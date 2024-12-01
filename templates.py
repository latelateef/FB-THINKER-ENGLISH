

def aspect_polarity_template(file_path = "./template_files/aspect_polarity_sentence.txt"):
    with open(file_path, 'r') as f:
        template = f.read()
    return template

def initial_summary_template(file_path = "./template_files/initial_summary.txt"):
    with open(file_path, 'r') as f:
        template = f.read()
    return template