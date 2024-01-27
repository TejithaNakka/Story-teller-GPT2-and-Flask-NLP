from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/genre')
def genre_selection():
    return render_template('genre_selection.html')

@app.route('/scary')
def scary_page():
    return render_template('scary.html')

@app.route('/humor')
def humor_page():
    return render_template('humor.html')

@app.route('/romance')
def romance_page():
    return render_template('romance.html')

if __name__ == '__main__':
    app.run(debug=True)
