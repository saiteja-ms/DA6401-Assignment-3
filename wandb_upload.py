import wandb

# Initialize a new run or use an existing one
run = wandb.init(project="transliteration-seq2seq", name="Connectivity_Visualization")

# Upload the PDF file
pdf_artifact = wandb.Artifact("connectivity_visualization", type="visualization")
pdf_artifact.add_file("connectivity_visualization.pdf")
run.log_artifact(pdf_artifact)
