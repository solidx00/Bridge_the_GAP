import os
import pickle
import re
import json

def extract_number(filename):
    """Extract the numeric value from a filename using a regex."""
    match = re.search(r'_(\d+)\.pkl$', filename)
    return int(match.group(1)) if match else float('inf')

def process_pkl_files(input_folder):
    """Process .pkl files in the input folder and generate JSON outputs."""
    combined_data = []
    filtered_data = []
    count_example = 0
    correct_count = 0
    multiple_selected_count = 0

    output_file = os.path.join(input_folder, 'numdoc5_retr5_template_info_all.json')
    filtered_output_file = os.path.join(input_folder, 'numdoc5_retr5_template_info_all_extended.json')

    # List and sort the .pkl files by number extracted from their names
    pkl_files = sorted([filename for filename in os.listdir(input_folder) if filename.endswith('.pkl')], key=extract_number)

    print("Trovati i seguenti file .pkl:")
    for filename in pkl_files:
        print(filename)

    # Read and combine data from all .pkl files
    for filename in pkl_files:
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            combined_data.extend(data)

    for example in combined_data:
        # Check if at least one response is correct
        is_correct = any(response.get('is_correct') for response in example.get('generated_responses', []))
        if is_correct:
            correct_count += 1

        # Check if there are multiple indices in 'selected_documents'
        selected_documents = example.get('selected_documents', [])
        if len(selected_documents) > 1:
            multiple_selected_count += 1

        # Find the best answer with the highest BLEU score
        best_answer = None
        best_bleu_score = 0
        for response in example.get('generated_responses', []):
            if response.get('is_correct') and response.get('bleu_score', 0) > best_bleu_score:
                best_bleu_score = response['bleu_score']
                best_answer = response.get('generated_answer', None)

        filtered_data.append({
            'query': example.get('query'),
            'are_answer': is_correct,
            'generated_answer': best_answer,
            'selected_documents': selected_documents,
            'number_documents_selected': len(selected_documents)
        })

        count_example += 1

    # Save combined data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    # Save filtered data to a JSON file
    with open(filtered_output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    return correct_count, multiple_selected_count, count_example

def main():
    """Main function to process .pkl files."""
    input_folder = input("Enter the path to the folder containing .pkl files: ").strip()
    if not os.path.isdir(input_folder):
        print("The specified folder does not exist.")
        return

    correct, multiple_selected, n_example = process_pkl_files(input_folder)

    print(f"Numeri di esempi totali: {n_example}")
    print(f"Esempi con almeno una risposta corretta: {correct}")
    print(f"Esempi con più di un indice in 'selected_documents': {multiple_selected}")

if __name__ == "__main__":
    main()