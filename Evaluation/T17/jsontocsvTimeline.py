import json
import csv

# Function to read JSONL file and extract data
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            for i in range(len(json_data)):
                date = json_data[i][0][0:10]
                sentences = json_data[i][1]
                data.append((date, sentences))
    return data

# Function to write data to CSV file
def write_to_csv(data, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Date', 'Sentences'])
        # Write data
        writer.writerows(data)

# Main function
def main():
    stories = ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]

    for i in range(len(stories)):
        jsonl_file_path = '/home/sojitra_2211mc15/Daivik/MDS/t17/'+stories[i]+'/timelines.jsonl'  # Replace with your file path
        csv_file_path = '/home/sojitra_2211mc15/Daivik/MDS/t17/tilseeval/Gold/outputTimelines'+stories[i]+'.csv'  # Replace with your desired output file path

        # Read data from JSONL file
        jsonl_data = read_jsonl(jsonl_file_path)

        # Write data to CSV file
        write_to_csv(jsonl_data, csv_file_path)
        

if __name__ == "__main__":
    main()
