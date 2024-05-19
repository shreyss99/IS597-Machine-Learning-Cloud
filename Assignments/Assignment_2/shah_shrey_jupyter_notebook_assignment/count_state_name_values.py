"""lists all unique state name values from input file and count of each."""

from csv import reader


def main():
    data_directory = 'data'
    input_filename = 'short_raw_data.csv'
    do_count_state_name_values(data_directory, input_filename)


def do_count_state_name_values(directory, filename):
    path_and_filename = f'{directory}/{filename}'
    infile = open(path_and_filename, 'r', encoding='utf-8')
    infile_reader = reader(infile)
    row_count = 0
    state_names = {}

    for row in infile_reader:
        row_count += 1
        if row_count > 1:
            state_name = row[1]
            state_names[state_name] = state_names.get(state_name, 0) + 1

    infile.close()

    key_values = list(state_names.keys())
    if len(key_values) < 1:
        print(f'No data records were found in {path_and_filename}.')
    else:
        key_values.sort()
        for key_value in key_values:
            print(f'{key_value}: {state_names[key_value]}')


if __name__ == '__main__':
    main()
