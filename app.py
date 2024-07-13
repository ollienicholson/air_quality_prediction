# app.py
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    
    # Handle the form submission logic here
    print(f"Received contact form submission from {name} ({email}): {message}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
    