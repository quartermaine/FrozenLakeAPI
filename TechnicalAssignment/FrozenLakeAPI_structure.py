from flask import Flask, jsonify, request
import gymnasium as gym
from uuid import uuid4
from typing import Dict, Any

class FrozenLakeAPI:
    def __init__(self) -> None:
        self.app: Flask = Flask(__name__)
        self.app.debug = True
        self.games: Dict[str, gym.Env] = {}

    def run_server(self) -> None:
        self.app.route('/new_game', methods=['POST'])(self.new_game)
        self.app.route('/step', methods=['POST'])(self.step)
        self.app.route('/reset', methods=['POST'])(self.reset)
        self.app.run(host="localhost", port=5005, threaded=True)

    def new_game(self) -> Any:
        """
        This method creates a FrozenLake environment and a unique token using uuid4. 
        The token is used to identify uniquely the environment so a client can specify
        at which one will send step and reset calls.

        Returns:
            Any: A JSON response with the client UUID and successful code or 
                an error and an unsuccessful code
        """
        #TODO
        try:
              env = gym.make('FrozenLake-v1', 
                             desc=None, 
                             map_name="4x4",
                             is_slippery=True
                               )

              game_id = str(uuid4())

              self.games[game_id] = env

              return jsonify({'game_id': game_id}), 200
        
        except Exception as e:

            return jsonify({'error': str(e)}), 500
        
        # raise NotImplementedError
    
    def reset(self) -> Any :
        """"
        This call should contain a uuid. 
        Resets the specified FrozenLake environment.

        Returns:
            Any: A JSON response with the observation from the environment reset and successful code or 
                an error and an unsuccessful code
        """
        #TODO
        try:
            data = request.get_json()

            game_id = data['game_id']

            if game_id not in self.games:

                return jsonify({'error': 'Invalid game_id'}), 400
            
            observation = self.games[game_id].reset()

            return jsonify({'observation': observation}), 200
        
        except Exception as e:

            return jsonify({'error': str(e)}), 500

        # raise NotImplementedError
    
    def step(self) -> Any :
        """
        This call should contain a uuid. 
        This call should contain a valid action. 
        Takes a step in the specified FrozenLake environment using the given action.

        Returns:
            Any: A JSON response with the observation, reward, done, truncated, and info and a successful code,
              or an error and an unsuccessful code.
        """
        #TODO

        try:
            data = request.get_json()

            game_id = data['game_id']

            action = data['action']

            if game_id not in self.games:

                return jsonify({'error': 'Invalid game_id'}), 400
            
            env = self.games[game_id]

            observation, reward, done, truncated, info = env.step(action)

            return jsonify({
                'observation': observation,
                'reward': reward,
                'done': done,
                'truncated': truncated,
                'info': info
            }), 200
        
        except Exception as e:
            
            return jsonify({'error': str(e)}), 500

        # raise NotImplementedError
    
if __name__ == '__main__':
    emulation_api:FrozenLakeAPI = FrozenLakeAPI()
    emulation_api.run_server()
