# File: examples/05_pii_protection.py

# SECTION 1: Imports
from dotenv import load_dotenv
from pathlib import Path

from psysafe.catalog.pii_protection.guardrail import PIIProtectionGuardrail
from psysafe.core.models import Conversation, Message
from psysafe.drivers.openai import OpenAIChatDriver

# Load environment variables
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"Warning: .env file not found at {ENV_PATH}. OpenAI API key might not be loaded.")

def main():
    """
    Main function to demonstrate the PIIProtectionGuardrail.
    """
    # SUBSECTION 2.1: Initialize Guardrail
    print("Initializing PIIProtectionGuardrail...")
    # Consider if this guardrail has parameters like sensitivity or specific PII types to focus on
    guardrail = PIIProtectionGuardrail()
    print("Guardrail initialized.")

    # SUBSECTION 2.1.1: Initialize and Bind LLM Driver
    print("Initializing LLM driver (OpenAIChatDriver)...")
    try:
        driver = OpenAIChatDriver(model="gpt-4o-mini")
        print("LLM driver initialized.")
        print("Binding driver to guardrail...")
        guardrail.bind(driver)
        print("Driver bound to guardrail.")
    except Exception as e:
        print(f"Error initializing or binding LLM driver: {e}")
        print("The script will likely fail at the check step without a functional driver.")
        return

    # SUBSECTION 2.2: Create Triggering Conversation (Contains PII)
    print("\nCreating a sample conversation designed to trigger the guardrail (Contains PII)...")
    triggering_messages_pii = [
        Message(role="user", content="Hi, I need to update my contact information."),
        Message(role="assistant", content="Sure, I can help with that. What is your email address and phone number?"),
        Message(role="user", content="My email is test.user@example.com and my phone is 0123-456-789. My full name is John Doe and I live at 123 Main St, Anytown.")
    ]
    triggering_conversation_pii = Conversation(messages=triggering_messages_pii)
    print("Triggering conversation (Contains PII) created.")

    # SUBSECTION 2.3: Create Non-Triggering Conversation (No PII)
    print("\nCreating a sample conversation designed NOT to trigger the guardrail (No PII)...")
    non_triggering_messages_no_pii = [
        Message(role="user", content="What are your opening hours?"),
        Message(role="assistant", content="We are open from 9 AM to 5 PM, Monday to Friday."),
        Message(role="user", content="Okay, thank you for the information.")
    ]
    non_triggering_conversation_no_pii = Conversation(messages=non_triggering_messages_no_pii)
    print("Non-triggering conversation (No PII) created.")

    # SUBSECTION 2.4: Process Triggering Conversation
    print("\nProcessing the triggering conversation (Contains PII)...")
    triggering_result_pii = guardrail.check(conversation=triggering_conversation_pii)
    print("Processing complete.")

    # SUBSECTION 2.5: Print Triggering Conversation Result
    # Expected fields in `details` for PIIProtectionGuardrail:
    # 'pii_detected': bool
    # 'pii_types': list of str (e.g., ['EMAIL', 'PHONE', 'ADDRESS', 'FULL_NAME'])
    # 'redaction_suggestions': list of dicts or str (detailing what to redact and how)
    # 'confidence_scores': dict (mapping PII type to confidence)
    print("\n--- Results for Triggering Conversation (Contains PII) ---")
    print(f"Guardrail Triggered (PII Detected): {triggering_result_pii.is_triggered}")
    print(f"PII Types Detected: {triggering_result_pii.details.get('pii_types', 'N/A')}")
    print(f"Redaction Suggestions: {triggering_result_pii.details.get('redaction_suggestions', 'N/A')}")
    print(f"Confidence Scores: {triggering_result_pii.details.get('confidence_scores', 'N/A')}")
    print(f"Full Details: {triggering_result_pii.details}")
    print(f"Errors: {triggering_result_pii.errors}")
    print(f"Metadata: {triggering_result_pii.metadata}")

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    print("\nProcessing the non-triggering conversation (No PII)...")
    non_triggering_result_no_pii = guardrail.check(conversation=non_triggering_conversation_no_pii)
    print("Processing complete.")

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    print("\n--- Results for Non-Triggering Conversation (No PII) ---")
    print(f"Guardrail Triggered (PII Detected): {non_triggering_result_no_pii.is_triggered}")
    print(f"PII Types Detected: {non_triggering_result_no_pii.details.get('pii_types', 'N/A')}")
    print(f"Redaction Suggestions: {non_triggering_result_no_pii.details.get('redaction_suggestions', 'N/A')}")
    print(f"Confidence Scores: {non_triggering_result_no_pii.details.get('confidence_scores', 'N/A')}")
    print(f"Full Details: {non_triggering_result_no_pii.details}")
    print(f"Errors: {non_triggering_result_no_pii.errors}")
    print(f"Metadata: {non_triggering_result_no_pii.metadata}")

# SECTION 3: Execute main function
if __name__ == "__main__":
    main()