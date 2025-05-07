from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json

from config import EVALUATION_RESULTS_PATH, EVALUATION_RESULTS_FILES_PATH

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    # Pass the correct path to your evaluation results
    correct_path = EVALUATION_RESULTS_FILES_PATH
    print(f"Passing evaluation results path to template: {correct_path}")
    return render_template('dashboard.html', evaluation_results_path=correct_path)


@app.route('/api/evaluations')
def list_evaluations():
    path = request.args.get('path', EVALUATION_RESULTS_FILES_PATH)
    print(f"API - Looking for files in: {path}")

    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return jsonify({'success': False, 'error': f'Path not found: {path}'})

    files = []
    for filename in os.listdir(path):
        if filename.endswith('.json') or filename.endswith('.csv'):
            files.append(os.path.join(path, filename))

    print(f"API - Found {len(files)} files: {files}")
    return jsonify({'success': True, 'files': files})


@app.route('/api/file')
def get_file():
    filepath = request.args.get('path', EVALUATION_RESULTS_PATH)
    print(f"API - Request to read file: {filepath}")

    if not filepath or not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return jsonify({'success': False, 'error': 'File not found'})

    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return jsonify(json.load(f))
    elif filepath.endswith('.csv'):
        with open(filepath, 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/plain'}
    else:
        return jsonify({'success': False, 'error': 'Unsupported file type'})


# Static file handling (for development)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    # Make sure the templates directory exists
    os.makedirs('templates', exist_ok=True)

    # Print configuration for debugging
    print(f"Using evaluation results path: {EVALUATION_RESULTS_FILES_PATH}")

    # Start the server
    app.run(debug=True, host='0.0.0.0', port=5000)
