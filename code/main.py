from clean_generator import generate_data
from noise import generate_noise

def main():

    # generates the clean data for both BPSK and QPSK modulations
    generate_data(10000, 1024)
    generate_noise()

if __name__ == "__main__":
    main()