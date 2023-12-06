import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def process_files(job_description, resume):
    with pdfplumber.open(job_description) as pdf:
        job_description_text = " ".join([page.extract_text() for page in pdf.pages])

    with pdfplumber.open(resume) as pdf:
        resume_text = " ".join([page.extract_text() for page in pdf.pages])

    content = [job_description_text, resume_text]

    cv = CountVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    match = similarity_matrix[0][1] * 100

    return match, job_description_text, resume_text


def about_us():
    st.title("About Us")
    st.subheader("Welcome to the About Us page!")
    st.write("Meet the creators of this application.")
    creators_info = {
        "Youssef Mouhyeddine": "Co-founder and developer",
        "Anas Radi": "Co-founder and designer",
        "Hamza El Bouamri": "Co-founder and data scientist"
    }

    for creator, description in creators_info.items():
        st.write(f"**{creator}**: {description}")


def contact_us():
    st.title("Contact Us")
    st.subheader("Use the form below to get in touch with us.")
    name = st.text_input("Your Name:")
    email = st.text_input("Your Email:")
    message = st.text_area("Your Message:")

    if st.button("Submit"):
        st.success("Form submitted successfully! We'll get back to you soon.")


def main():
    nav_choice = st.sidebar.radio("Navigation", ["Home", "Upload CV", "About Us", "Contact Us"])

    if nav_choice == "Home":
        st.title("CVCV")
        st.write("Welcome to the CVCV application! This tool is designed to analyze resumes and job descriptions "
                 "to determine the match percentage between a candidate and a job. It is part of an educational project "
                 "suggested by the school for the end of the year.")

        st.write("The primary goal is to check whether a candidate is qualified for a role based on their education, "
                 "experience, and other information captured in their resume. It uses natural language processing (NLP) "
                 "techniques for pattern matching.")

        col1, col2 = st.columns(2)
        with col2:
            st.image("images/home.png", width=400, output_format="PNG")


    elif nav_choice == "Upload CV":
        st.subheader("CVCV Analyzer")
        uploadedJD = st.file_uploader("Upload Job Description (PDF)", type="pdf")
        uploadedResume = st.file_uploader("Upload CV (PDF)", type="pdf")

        click = st.button("Process")

        if click and uploadedJD:
            job_description_match, job_description_text, resume_text = process_files(uploadedJD, uploadedResume)

            job_description_match = round(job_description_match, 2)

            st.write(f"Match Percentage: {job_description_match}%")

            if job_description_match >= 70:
                st.success("Congratulations! The candidate is a good match for the job.")
            else:
                st.warning("The candidate may not be an ideal match for the job. Consider reviewing the criteria.")

            st.subheader("Extracted Text:")
            st.write("Resume:")
            st.text(resume_text)

    elif nav_choice == "About Us":
        about_us()

    elif nav_choice == "Contact Us":
        contact_us()


if __name__ == "__main__":
    main()
