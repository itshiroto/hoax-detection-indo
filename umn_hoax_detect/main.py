from umn_hoax_detect.data.loader import load_dataset


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} hoax entries.")
    print(df.head(3))


if __name__ == "__main__":
    main()
