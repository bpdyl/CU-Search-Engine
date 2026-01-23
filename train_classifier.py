"""Script to train the document classifier with expanded dataset."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.trainer import ClassifierTrainer, save_sample_training_data
from collections import Counter

def main():
    print("="*60)
    print("DOCUMENT CLASSIFIER TRAINING")
    print("="*60)

    # Save the expanded training data
    print("\n1. Saving expanded training data...")
    save_sample_training_data()

    # Train the classifier
    print("\n2. Training classifier with expanded dataset...")
    trainer = ClassifierTrainer()
    texts, labels = trainer.load_training_data()
    print(f"   Loaded {len(texts)} training samples")

    # Count by category
    counts = Counter(labels)
    print("\n   Samples per category:")
    for cat, count in sorted(counts.items()):
        print(f"     - {cat}: {count}")

    # Train
    print("\n3. Training model...")
    stats = trainer.train(texts, labels)

    # Print report
    print("\n" + trainer.get_training_report())

    # Save model
    print("\n4. Saving model...")
    trainer.save_model()
    print("\nTraining complete!")

    # Test with "stranger things"
    print("\n" + "="*60)
    print("TESTING WITH SHORT QUERIES")
    print("="*60)

    from classifier.predictor import DocumentClassifier
    classifier = DocumentClassifier()

    test_queries = [
        "stranger things",
        "Stranger Things Netflix",
        "Taylor Swift concert",
        "stock market crash",
        "cancer treatment",
        "Game of Thrones",
        "Federal Reserve interest rate",
        "mental health awareness",
        "Marvel Avengers movie",
        "COVID-19 vaccine"
    ]

    print("\nTest Results:")
    print("-"*60)
    for query in test_queries:
        result = classifier.classify(query)
        print(f"'{query}'")
        print(f"  -> {result['category']} ({result['confidence']*100:.1f}% confidence)")
        print()

if __name__ == "__main__":
    main()
