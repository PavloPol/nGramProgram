import csv
from io import BytesIO
import json
from PyPDF2 import PdfReader
from flask import Flask, request, render_template, send_file
from collections import Counter
from typing import List, Tuple
from docx import Document
import os
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to save uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# N-gram analysis functions
def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    words = text.split()
    return words


def generate_ngrams(words: List[str], n: int) -> List[Tuple[str]]:
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return ngrams


def analyze_text(text: str, n: int) -> Counter:
    words = preprocess_text(text)
    ngrams = generate_ngrams(words, n)
    return Counter(ngrams)


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def read_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)


def read_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return "\n".join([" ".join(row) for row in reader])


@app.route("/", methods=["GET", "POST"])
def ngram_analysis():
    text = ""
    n = 2
    results = None
    error_message = None

    if request.method == "POST":
        text = request.form.get("text", "")

        # Handle file upload
        file = request.files.get("file")
        if file and file.filename:
            file_ext = file.filename.split('.')[-1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            try:
                if file_ext == "txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                elif file_ext == "docx":
                    text = read_docx(file_path)
                elif file_ext == "pdf":
                    text = read_pdf(file_path)
                elif file_ext == "csv":
                    text = read_csv(file_path)
                else:
                    error_message = "Unsupported file type. Please upload a .txt, .docx, .pdf, or .csv file."
            except Exception as e:
                error_message = f"Error reading file: {e}"

        # Handle N input
        try:
            n = int(request.form.get("n", 2))
            if n < 2 or n > 7:
                raise ValueError("N must be between 2 and 7.")
        except ValueError as e:
            error_message = str(e)
            n = 2  # Default value

        # Process the text if no errors
        if text.strip() and not error_message:
            ngram_counts = analyze_text(text, n)
            results = [(f"{' '.join(ngram)}", count) for ngram, count in ngram_counts.items()]

    return render_template(
        "index.html",
        text=text,
        n=n,
        results=results,
        results_json=json.dumps(results) if results else None,  # Serialize results for download
        error_message=error_message
    )

@app.route("/download", methods=["POST"])
def download_csv():
    ngram_data = request.form.get("results_json")
    if not ngram_data:
        return "No data available to download", 400

    try:
        ngram_data = json.loads(ngram_data)  # Deserialize JSON
    except json.JSONDecodeError:
        return "Error processing data", 400

    # Prepare CSV
    output = BytesIO()
    # Write CSV with UTF-8 encoding and BOM
    output.write("\ufeff".encode("utf-8"))  # Add BOM for UTF-8
    text_output = io.TextIOWrapper(output, encoding="utf-8", newline="")
    try:
        writer = csv.writer(text_output)
        writer.writerow(["N-gram", "Count"])
        writer.writerows(ngram_data)
        text_output.flush()  # Flush the TextIOWrapper's buffer to the BytesIO object
    finally:
        text_output.detach()  # Detach the TextIOWrapper without closing BytesIO

    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="ngram_results.csv"
    )


if __name__ == "__main__":
    app.run(debug=True)
