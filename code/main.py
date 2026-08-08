from clean_generator import generate_data
from noise import generate_noise

def main():
    split_config = {
            "train": 10000,
            "test": 3000,
            "validation": 2000
            }
    # generates the clean data for both BPSK and QPSK modulations
    generate_data(10000, 1024, split_config)
    generate_noise(split_config)

if __name__ == "__main__":
    main()