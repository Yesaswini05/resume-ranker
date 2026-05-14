from flask import Flask, render_template, request, send_file
import os
import pdfplumber
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create resumes folder automatically
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Store latest scores
latest_scores = []


# Extract text from PDF
def extract_text(file_path):

    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                content = page.extract_text()

                if content:
                    text += content

        return text.lower()

    except Exception as e:

        print("PDF Error:", e)
        return ""


# Home Page
@app.route('/', methods=['GET', 'POST'])
def index():

    global latest_scores

    scores = []

    if request.method == 'POST':

        job_desc = request.form['job_desc']

        files = request.files.getlist('resumes')

        resume_texts = []
        file_names = []

        for file in files:

            if file.filename != "":

                path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    file.filename
                )

                file.save(path)

                text = extract_text(path)

                print("EXTRACTED TEXT:")
                print(text[:300])

                resume_texts.append(text)

                file_names.append(file.filename)

        # Avoid empty resumes
        if len(resume_texts) > 0:

            documents = [job_desc.lower()] + resume_texts

            tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2)
            )

            vectors = tfidf.fit_transform(documents)

            similarity = cosine_similarity(
                vectors[0:1],
                vectors[1:]
            ).flatten()

            scores = list(zip(file_names, similarity))

            scores = sorted(
                scores,
                key=lambda x: x[1],
                reverse=True
            )

            scores = [
                (name, round(score * 100, 2))
                for name, score in scores
            ]

            latest_scores = scores

    return render_template(
        'index.html',
        scores=scores
    )


# Download CSV Report
@app.route('/download')
def download_report():

    global latest_scores

    df = pd.DataFrame(
        latest_scores,
        columns=['Resume Name', 'Score (%)']
    )

    report_path = 'resume_report.csv'

    df.to_csv(report_path, index=False)

    return send_file(
        report_path,
        as_attachment=True
    )


# Run Flask App
if __name__ == '__main__':

    app.run(debug=True)