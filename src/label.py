import csv
import os

INPUT_FILE = "data/raw_events.csv"
OUTPUT_FILE = "data/training_data.csv"

def load_existing_labels():
    labeled_titles = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled_titles.add(row["title"])
    return labeled_titles

def main():
    with open(INPUT_FILE, newline="") as f:
        reader = csv.DictReader(f)
        events = list(reader)

    labeled_titles = load_existing_labels()
    remaining = [e for e in events if e["title"] not in labeled_titles]

    print(f"Total events: {len(events)}")
    print(f"Already labeled: {len(labeled_titles)}")
    print(f"Remaining: {len(remaining)}")
    print("Type 1 if relevant, 0 if not, or q to quit and save.\n")

    file_exists = os.path.exists(OUTPUT_FILE)
    fieldnames = ["title", "date", "location", "format", "description", "relevant"]

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, event in enumerate(remaining):
            print(f"\n[{i+1}/{len(remaining)}] {event['title']}")
            print(f"  Date: {event['date']}")
            print(f"  Location: {event['location']}")
            print(f"  Format: {event['format']}")
            print(f"  Description: {event['description'][:150]}")

            answer = input("Relevant? (1/0/q): ").strip().lower()

            if answer == "q":
                print("\nSaved progress. Run again to continue.")
                return

            if answer not in ("1", "0"):
                print("Invalid input, skipping this one for now.")
                continue

            event["relevant"] = answer
            writer.writerow(event)
            f.flush()

    print("\nAll events labeled! Saved to data/training_data.csv")

if __name__ == "__main__":
    main()