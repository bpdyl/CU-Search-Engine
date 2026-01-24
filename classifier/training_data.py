"""
Training Data for Document Classification

Loads labeled documents for Business, Entertainment, and Health categories from CSV files.
Includes long-form articles, medium-length summaries, and short-form queries/phrases
to enable the classifier to handle various text lengths effectively.

Data is stored in CSV files located in the training_data/ directory:
- business_training_data.csv
- entertainment_training_data.csv
- health_training_data.csv

CSV Format:
- text: The training text content
- category: Business, Entertainment, or Health
- length_type: Long, Mid, or Short

Documents are inspired by and representative of content from public news sources
including BBC News, Reuters, and other major news outlets.
"""
import os
import csv


# Get the directory where this module is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(MODULE_DIR, 'training_data')


def load_csv_data(filepath):
    """
    Load training data from a CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        List of dictionaries with 'text', 'category', and 'length_type' keys
    """
    samples = []

    if not os.path.exists(filepath):
        print(f"Warning: Training data file not found: {filepath}")
        return samples

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('text') and row.get('category'):
                samples.append({
                    'text': row['text'].strip(),
                    'category': row['category'].strip(),
                    'length_type': row.get('length_type', 'Long').strip()
                })

    return samples


def get_training_data():
    """
    Get comprehensive training data for the classifier.

    Loads data from CSV files in the training_data/ directory.

    Returns:
        List of dictionaries with 'text' and 'category' keys
    """
    samples = []

    # Define CSV files to load
    csv_files = [
        os.path.join(TRAINING_DATA_DIR, 'business_training_data.csv'),
        os.path.join(TRAINING_DATA_DIR, 'entertainment_training_data.csv'),
        os.path.join(TRAINING_DATA_DIR, 'health_training_data.csv'),
    ]

    # Load data from each CSV file
    for csv_file in csv_files:
        file_samples = load_csv_data(csv_file)
        samples.extend(file_samples)

    # Convert to the expected format (text and category only)
    return [{'text': s['text'], 'category': s['category']} for s in samples]


def get_training_data_with_length():
    """
    Get comprehensive training data with length type information.

    Returns:
        List of dictionaries with 'text', 'category', and 'length_type' keys
    """
    samples = []

    csv_files = [
        os.path.join(TRAINING_DATA_DIR, 'business_training_data.csv'),
        os.path.join(TRAINING_DATA_DIR, 'entertainment_training_data.csv'),
        os.path.join(TRAINING_DATA_DIR, 'health_training_data.csv'),
    ]

    for csv_file in csv_files:
        file_samples = load_csv_data(csv_file)
        samples.extend(file_samples)

    return samples


def get_data_statistics(samples=None):
    """
    Get statistics about the training data.

    Args:
        samples: Optional list of samples. If None, loads from CSV files.

    Returns:
        Dictionary with total count and breakdown by category and length type
    """
    if samples is None:
        samples = get_training_data_with_length()

    stats = {
        "total": len(samples),
        "by_category": {},
        "by_length_type": {},
        "by_category_and_length": {}
    }

    for sample in samples:
        cat = sample.get("category", "Unknown")
        length = sample.get("length_type", "Unknown")

        # Count by category
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        # Count by length type
        stats["by_length_type"][length] = stats["by_length_type"].get(length, 0) + 1

        # Count by category and length
        key = f"{cat}_{length}"
        stats["by_category_and_length"][key] = stats["by_category_and_length"].get(key, 0) + 1

    return stats


def print_data_summary():
    """Print a summary of the training data."""
    stats = get_data_statistics()

    print("\n" + "=" * 50)
    print("Training Data Summary")
    print("=" * 50)
    print(f"\nTotal samples: {stats['total']}")

    print("\nBy Category:")
    for cat, count in sorted(stats['by_category'].items()):
        print(f"  {cat}: {count}")

    print("\nBy Length Type:")
    for length, count in sorted(stats['by_length_type'].items()):
        print(f"  {length}: {count}")

    print("\nBy Category and Length:")
    for key, count in sorted(stats['by_category_and_length'].items()):
        print(f"  {key}: {count}")
    print()


if __name__ == '__main__':
    # When run directly, print statistics
    print_data_summary()
