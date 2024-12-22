import PyPDF2


class TextbookProcessor:
    def extract_content(self, textbook_path):
        content = ""
        with open(textbook_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                content += page.extract_text()
        return content

    def preprocess_content(self, content):
        # Implement preprocessing steps (e.g., removing special characters, lowercasing)
        # This is a placeholder implementation
        return content.lower()
