from ingestion import ingest_and_merge_pdf

if __name__ == "__main__":
    # Change this to an actual PDF path on your server
    MY_PDF = "/home/lisa/Arupreza/Agents/9.LG_CorrectiveRAG/A_Survey_of_Cybersecurity_Challenges_and_Mitigation_Techniques_for_Connected_and_Autonomous_Vehicles.pdf"
    ingest_and_merge_pdf(MY_PDF)