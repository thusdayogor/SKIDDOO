import os
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import native_bayes
from threading import Thread
from queue import Queue

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skiddoo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Websites(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    url_site = db.Column(db.String(255), nullable = False)
    title =  db.Column(db.String(255), nullable = False)
    label = db.Column(db.String(255), nullable=False)
    danger = db.Column(db.String(255),nullable = False)



@app.route('/')


def index():

    input = request.args.get('input')

    if input:
        websites = Websites.query.filter(Websites.title.contains(input) | Websites.label.contains(input))
        return render_template('search.html', data = websites)

    return render_template('index.html')


@app.route('/search')

def search():

    input = request.args.get('input')
    if input:
        websites = Websites.query.filter(Websites.title.contains(input) | Websites.label.contains(input))
        return render_template('search.html', data = websites)


    return render_template('search.html')



filename = "classifier/check.csv"

def transformation(predict):
    if predict == 1:
        return "The site contains inappropriate content!"
    else:
        return "The site contains valid content"
    return "Result is undefined"

def native(url_site,queue):
    web_content = native_bayes.get_html_content(url_site)
    clean_from_tags = native_bayes.cleaning_text_from_tags(web_content.content)
    clean_from_numbers = native_bayes.cleaning_numbers(clean_from_tags)
    clean_from_mark = native_bayes.cleaning_punction_mark(clean_from_numbers)
    clean = native_bayes.cleaning_stop_words(clean_from_mark)
    native_bayes.write_to_csv(filename,url_site,clean)
    predict = native_bayes.special_model(filename)
    result = transformation(predict)
    os.remove(filename)
    queue.put(result)



@app.route('/add-site', methods = ['POST', 'GET'])

def add_site():

    if request.method =='POST':
        title = request.form['title']
        url_site = request.form['url_site']
        label = request.form['label']

        queue = Queue()
        th = Thread(target=native,args=(url_site,queue))
        th.start()
        th.join()
        danger = queue.get()
        print("Result of danger:",danger)


        websites = Websites(title = title, url_site = url_site,label = label,danger = danger)


        try:
            db.session.add(websites)
            db.session.commit()
            return redirect('/')
        except:
            return 'ERROR 401'

    return render_template('add-site.html')


if __name__ == "__main__":
    app.run(debug=True)




