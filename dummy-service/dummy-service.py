#!/usr/bin/env python3

from flask import Flask, flash, redirect, render_template, request, session


app = Flask(__name__)


@app.route('/')
def root():
    return page('landing page')


@app.route('/<path>')
def page(path):
    if not session.get('username'):
        return render_template('login.html')
    page = path.replace('_', ' ').replace('-', ' ').title()
    return f'{page} (user {session["username"]})'


@app.route('/login', methods=['POST'])
def login():
    if request.form['username'] and request.form['password']:
        session['username'] = request.form['username']
        return redirect('/start-page')
    return render_template('login.html')


if __name__ == "__main__":
    app.secret_key = 'Z15321255'
    app.run(debug=True, host='0.0.0.0')
