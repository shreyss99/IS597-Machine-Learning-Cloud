"""
Reads raw data file and writes data file with city coding errors cleaned.
"""

from csv import reader


def main():
    data_directory = 'data'
    input_filename = 'raw_data.csv'
    output_filename = 'cleaned_data.csv'
    do_clean_data_coding_errors(data_directory, input_filename, output_filename)


def do_clean_data_coding_errors(directory, infile_name, outfile_name):
    infile_path_and_name = f'{directory}/{infile_name}'
    outfile_path_and_name = f'{directory}/{outfile_name}'
    infile = open(infile_path_and_name, 'r', encoding='utf-8')
    the_reader = reader(infile)
    outfile = open(outfile_path_and_name, 'w', encoding='utf-8')
    print('City,State,Quantity', file=outfile)
    row_number = 0
    records_processed = 0

    for row in the_reader:
        row_number += 1
        if row_number > 1:
            city_name = row[0]
            state_name = row[1]
            quantity = row[2]
            fixed_city_name = city_name.strip().title()
            fixed_state_name = state_name.strip().title()
            output_line = f'{fixed_city_name},{fixed_state_name},{quantity}'
            print(output_line, file=outfile)
            records_processed += 1

    infile.close()
    outfile.close()
    print(f'{records_processed} cleaned records were written to {outfile_path_and_name}.')


if __name__ == '__main__':
    main()
