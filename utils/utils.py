def print_color_and_alert(text, value, low_thresh1,low_thresh2, high_thresh1, high_thresh2):
    if value < low_thresh1:
        print(f'\033[37m\033[41m'+f'{text} is {value:.4f} too low...'+'\033[0m')
    elif low_thresh1 <= value < low_thresh2:
        print(f'\033[37m\033[43m'+f'{text} is {value:.4f} bit low...'+'\033[0m')
    elif low_thresh2 <= value < high_thresh1:
        print(f'\033[37m\033[42m'+f'{text} is {value:.4f} OK'+'\033[0m')
    elif high_thresh1 <= value < high_thresh2:
        print(f'\033[37m\033[43m'+f'{text} is {value:.4f} bit high...'+'\033[0m')
    elif high_thresh2 <= value:
        print(f'\033[37m\033[41m'+f'{text} is {value:.4f} too high...'+'\033[0m')