from clean_generator import generate_data
from noise import generate_noise

def main():
    split_config = {
            "train": 10000,
            "test": 3000,
            "validation": 2000
            }
    # generates the clean data for both BPSK and QPSK modulations
    generate_data(signal_length = 1024, sps = 8, rrc_beta = 0.35, rrc_span = 8, split_config = {"train": 10000, "test": 3000, "validation": 2000})
    #generate_noise(split_config)

if __name__ == "__main__":

    main()

