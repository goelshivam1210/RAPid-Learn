import math
import argparse

#Argparser for evaluation settings (Connected to Trade, NG, independent jsons, tournament manager, etc.)

ap = argparse.ArgumentParser()
ap.add_argument("-E", "--env", default='polycraft', type=str, required=False, help="Env to perform evaluation on")
ap.add_argument("--disable_trade", dest='trade', default=True, required=False, action='store_false', help="Connect to Trade or communicate directly with env")
ap.add_argument("--disable_manager", dest='manager', default=True, required=False, action='store_false', help="Using PAL tournament manager to run the evaluation")
ap.add_argument("-H", "--host", default='127.0.0.1', type=str, required=False, help="Host for socket connection")
ap.add_argument("-TP", "--trade_port", default=6013, type=int, required=False, help="Port for trade socket connection")
ap.add_argument("-EP", "--env_port", default=9000, type=int, required=False, help="Port for env socket connection")
ap.add_argument("-R", "--reset_command", default='RESET domain ../available_tests/pogo_nonov.json', type=str, required=False, help="command for initial polycraft init/reset")
ap.add_argument("-S", "--stepcost", default=10000000, type=int, required=False, help="Max step cost per trial")
ap.add_argument("-T", "--time", default=6000, type=int, required=False, help="Max time per trial")
ap.add_argument("-J", "--jsons_dir", default='', type=str, required=False, help="directory containing trial jsons for independent polycraft testing case")

#debugging params
ap.add_argument("--stepcost_penalty", default=0.012, type=float, required=False, help="neg reward incurred per stepcost")
# ap.add_argument("--restrict_actions", dest='restrict', default=False, required=False, action='store_true', help="Using PAL tournament manager to run the evaluation")
ap.add_argument("--restrict_beneficial", dest='restrict_beneficial', default=False, required=False, action='store_true', help="Using PAL tournament manager to run the evaluation")

#TODO: novelty args if using NG

args = vars(ap.parse_args())

#PIPENV ONLY
##Polycraft Tournament Manager no trade: sudo PYTHONPATH=$PYTHONPATH /home/dev/.local/share/virtualenvs/polycraft_tufts-_CYaxy1P/bin/python LaunchTournament.py
#       In config file: AGENT_COMMAND_UNIX = "/home/dev/.local/share/virtualenvs/polycraft_tufts-_CYaxy1P/bin/python simulate_tournament.py --disable_trade"
##Polycraft Independent Jsons: sudo PYTHONPATH=$PYTHONPATH /home/dev/.local/share/virtualenvs/polycraft_tufts-_CYaxy1P/bin/python simulate_tournament.py --disable_trade --disable_manager -T 600 -S 1000000 -J /home/dev/tufts/polycraft_tufts/rl_agent/dqn_lambda/novelty_jsons/2021.03.11_POGO_tournament/POGO_100game_shared_novelties/POGO_L01_T02_S01/X0100/POGO_L01_T02_S01_X0100_H_U0019_V1/
##NG: sudo PYTHONPATH=$PYTHONPATH /home/dev/.local/share/virtualenvs/polycraft_tufts-_CYaxy1P/bin/python simulate_tournament.py --disable_trade --disable_manager -E novelgridworld -T 600 -S 1000000