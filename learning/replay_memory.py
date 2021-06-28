import numpy as np
import re

from polycraft_tufts.rl_agent.dqn_lambda.learning.return_calculation import calculate_lambda_returns, calculate_nstep_returns


# Pengs-Median seems like best general approach if we cannot tune Lambda although takes longer
# Otherwise watkins-0.99 is quicker but may not be ideal for every task (although with same reward signal for all
#   subtasks, maybe it would work well - would have to test on more novelties)
def make_replay_memory(return_est, capacity, history_len, discount, cache_size, block_size, priority):
    shared_args = (capacity, history_len, discount, cache_size, block_size, priority)
    int_capture = r'([0-9]+)'
    float_capture = r'([0-9]+\.[0-9]+)'

    match = re.match('nstep-' + int_capture, return_est)
    if match:
        n = int(match.group(1))
        return NStepReplayMemory(*shared_args, n)

    match = re.match('pengs-' + float_capture, return_est)
    if match:
        lambd = float(match.group(1))
        return LambdaReplayMemory(*shared_args, lambd, use_watkins=False)

    match = re.match('watkins-' + float_capture, return_est)
    if match:
        lambd = float(match.group(1))
        return LambdaReplayMemory(*shared_args, lambd, use_watkins=True)

    if return_est == 'pengs-median':
        return MedianLambdaReplayMemory(*shared_args, use_watkins=False)

    if return_est == 'watkins-median':
        return MedianLambdaReplayMemory(*shared_args, use_watkins=True)

    match = re.match('pengs-maxtd-' + float_capture, return_est)
    if match:
        max_td = float(match.group(1))
        return MeanSquaredTDLambdaReplayMemory(*shared_args, max_td, use_watkins=False)

    match = re.match('watkins-maxtd-' + float_capture, return_est)
    if match:
        max_td = float(match.group(1))
        return MeanSquaredTDLambdaReplayMemory(*shared_args, max_td, use_watkins=True)

    raise ValueError('Unrecognized return estimator {}'.format(return_est))


class ReplayMemory:
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority):
        assert (cache_size % block_size) == 0
        # Extra samples to fit exactly `capacity` (overlapping) blocks
        self.capacity = capacity + (history_len - 1) + block_size
        self.history_len = history_len
        self.discount = discount
        self.num_samples = 0

        self.cache_size = cache_size
        self.block_size = block_size
        self.priority = priority
        self.refresh_func = None

        # Main variables for memory
        self.obs = None  # Allocated dynamically once shape/dtype are known
        self.actions = np.empty([self.capacity], dtype=np.int32)
        self.rewards = np.empty([self.capacity], dtype=np.float32)
        self.dones = np.empty([self.capacity], dtype=np.bool)
        self.next = 0  # Points to next transition to be overwritten

        # Auxiliary buffers for the cache -- pre-allocated to smooth memory usage
        self.cached_states  = None  # Allocated dynamically once shape/dtype are known
        self.cached_actions = np.empty([self.cache_size], dtype=np.int32)
        self.cached_returns = np.empty([self.cache_size], dtype=np.float32)
        self.cached_errors  = np.empty([self.cache_size], dtype=np.float32)
        self.cached_indices = np.empty([self.cache_size], dtype=np.int32)

    def register_refresh_func(self, f):
        assert self.refresh_func is None
        self.refresh_func = f

    def sample(self, batch_size):
        start = self.batch_counter * batch_size
        end = start + batch_size
        indices = self.cached_indices[start:end]

        state_batch = self.cached_states[indices]
        action_batch = self.cached_actions[indices]
        return_batch = self.cached_returns[indices]

        self.batch_counter += 1

        return np.array(state_batch), np.array(action_batch), np.array(return_batch)

    def encode_recent_observation(self):
        i = self.len()
        return self._encode_observation(i)

    def _encode_observation(self, i):
        i = self._align(i)

        # Start with blank observations except the last
        state = np.zeros([self.history_len, *self.obs[0].shape], dtype=self.obs[0].dtype)
        state[-1] = self.obs[i]

        # Fill-in backwards, break if we reach a terminal state
        for j in range(1, min(self.history_len, self.len())):
            if self.dones[i-j]:
                break
            state[-1-j] = self.obs[i-j]

        return state

    def _align(self, i):
        # Make relative to pointer when full
        if not self.full(): return i
        return (i + self.next) % self.capacity

    def store_obs(self, obs):
        if self.obs is None:
            self.obs = np.empty([self.capacity, *obs.shape], dtype=obs.dtype)
        if self.cached_states is None:
            self.cached_states = np.empty([self.cache_size, self.history_len, *obs.shape], dtype=obs.dtype)
        self.obs[self.next] = obs

    def store_effect(self, action, reward, done):
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.dones[self.next] = done

        self.next = (self.next + 1) % self.capacity
        self.num_samples = min(self.capacity, self.num_samples + 1)

    def len(self):
        return self.num_samples

    def full(self):
        return self.len() == self.capacity

    def refresh(self, train_frac):
        # Reset batch counter
        self.batch_counter = 0

        # Sample blocks until we have enough data
        num_blocks = self.cache_size // self.block_size
        block_ids = self._sample_block_ids(num_blocks)

        self._refresh(train_frac, block_ids)  # Separate function for unit testing

    def _refresh(self, train_frac, block_ids):
        # Refresh the blocks we sampled and load them into the cache
        for k, i in enumerate(block_ids):
            states = self._extract_block(None, i, states=True)
            actions = self._extract_block(self.actions, i)
            rewards = self._extract_block(self.rewards, i)
            dones = self._extract_block(self.dones, i)

            max_qvalues, mask, onpolicy_qvalues = self.refresh_func(states, actions)
            returns = self._calculate_returns(rewards, max_qvalues, dones, mask)
            errors = np.abs(returns - onpolicy_qvalues)

            start = self.block_size * k
            end = start + self.block_size

            self.cached_states[start:end] = states[:-1]
            self.cached_actions[start:end] = actions
            self.cached_returns[start:end] = returns
            self.cached_errors[start:end] = errors

        # Prioritize samples
        distr = self._prioritized_distribution(self.cached_errors, train_frac)
        self.cached_indices = np.random.choice(self.cache_size, size=self.cache_size, replace=True, p=distr)

    def _sample_block_ids(self, n):
        return np.random.randint(self.history_len - 1, self.len() - self.block_size, size=n)

    def _extract_block(self, a, start, states=False):
        end = start + self.block_size
        if states:
            assert a is None
            return np.array([self._encode_observation(i) for i in range(start, end + 1)])
        return a[self._align(np.arange(start, end))]

    def _prioritized_distribution(self, errors, train_frac):
        # Start with the uniform distribution.
        distr = np.ones_like(errors) / self.cache_size
        # Adjust the probabilities based on whether their corresponding errors lie above/below the median.
        p = self.priority_now(train_frac)
        med = np.median(errors)
        distr[errors > med] *= (1.0 + p)
        distr[errors < med] *= (1.0 - p)
        # Note that if the error was identically equal to the median, its probability was not adjusted;
        # this is the correct behavior to guarantee the probabilities sum to 1.
        # However, due to floating point errors, we still need to re-normalize the distribution here:
        return distr / distr.sum()

    def priority_now(self, train_frac):
        return self.priority * (1.0 - train_frac)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        raise NotImplementedError


class LambdaReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins):
        self.lambd = lambd
        self.use_watkins = use_watkins
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        return calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, self.lambd)


class MedianLambdaReplayMemory(LambdaReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, use_watkins):
        lambd = None
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins)

    def _calculate_returns(self, rewards, qvalues, dones, mask, k=21):
        if not self.use_watkins:
            mask = np.ones_like(qvalues)
        assert k > 1
        returns = np.empty(shape=[k, rewards.size], dtype=np.float32)
        for i in range(0, k):
            returns[i] = calculate_lambda_returns(rewards, qvalues, dones, mask, self.discount, lambd=i/(k-1))
        return np.median(returns, axis=0)


class MeanSquaredTDLambdaReplayMemory(LambdaReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, max_td, use_watkins):
        lambd = None
        self.max_td = max_td
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority, lambd, use_watkins)

    def _calculate_returns(self, rewards, qvalues, dones, mask, k=7):
        f = super()._calculate_returns  # Use parent function to compute returns

        # Try the extremes first
        returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd=1.0)
        if ok:
            return returns

        returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd=0.0)
        if not ok:
            return returns

        # If we haven't returned by now, we need to search for a good lambda value
        best_returns = None
        lambd = 0.5

        for i in range(2, 2 + k):
            returns, ok = self._try_lambda(f, rewards, qvalues, dones, mask, lambd)

            if ok:
                best_returns = returns
                lambd += 1.0 / (2.0 ** i)
            else:
                lambd -= 1.0 / (2.0 ** i)

        return best_returns if best_returns is not None else returns

    def _try_lambda(self, f, rewards, qvalues, dones, mask, lambd):
        self.lambd = lambd  # Pass implicitly to parent function
        returns = f(rewards, qvalues, dones, mask)
        td_error = np.square(returns - qvalues[:-1]).mean()
        ok = (td_error <= self.max_td)
        return returns, ok


class NStepReplayMemory(ReplayMemory):
    def __init__(self, capacity, history_len, discount, cache_size, block_size, priority, n):
        self.n = n
        super().__init__(capacity, history_len, discount, cache_size, block_size, priority)

    def _calculate_returns(self, rewards, qvalues, dones, mask):
        return calculate_nstep_returns(rewards, qvalues, dones, self.discount, self.n)
