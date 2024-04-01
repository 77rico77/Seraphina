from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/avatar', methods=['POST'])
def create_avatar():
    if request.method == 'POST':
        # Extracting body data
        data = request.json
        
        # Base command
        cmd = ['python', 'inference.py']
        
        # Parameters that should be included only if they are provided
        param_keys = [
            'source_image', 'driven_audio', 'result_dir', 'preprocess', 'pose_style',
            'input_yaw', 'input_pitch', 'input_roll', 
            'ref_eyeblink', 'ref_pose'
        ]
        
        # Building the command by iterating over the parameters
        for key in param_keys:
            value = data.get(key)
            if value is not None:  # This checks if the parameter exists
                print(key, value)
                cmd += [f'--{key}', str(value)]
        
        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            # Respond with status 200
            return jsonify({"message": "Inference process started successfully"}), 200
        except subprocess.CalledProcessError as e:
            # Handle errors in the subprocess
            return jsonify({"error": "Inference process failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
