# File: examples/05_pii_protection.py

# SECTION 1: Imports
from utils.llm_utils import load_environment # Changed import
from psysafe.catalog.pii_protection.guardrail import PiiProtectionGuardrail # Corrected import
from psysafe.core.models import Conversation, Message
from psysafe.drivers.openai import OpenAIChatDriver

load_environment() # Added environment loading call

def main():
    """
    Main function to demonstrate the PIIProtectionGuardrail.
    """
    # SUBSECTION 2.1: Initialize Guardrail
    print("Initializing PIIProtectionGuardrail...")
    # Consider if this guardrail has parameters like sensitivity or specific PII types to focus on
    guardrail = PiiProtectionGuardrail() # Corrected class name casing
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
    triggering_messages = [ # Renamed variable
        Message(role="user", content="Hi, I need to update my contact information."),
        Message(role="assistant", content="Sure, I can help with that. What is your email address and phone number?"),
        Message(role="user", content="My email is test.user@example.com and my phone is 0123-456-789. My full name is John Doe and I live at 123 Main St, Anytown.")
    ]
    triggering_conversation = Conversation(messages=triggering_messages) # Renamed variable
    print("Triggering conversation (Contains PII) created.")

    # SUBSECTION 2.3: Create Non-Triggering Conversation (No PII)
    print("\nCreating a sample conversation designed NOT to trigger the guardrail (No PII)...")
    non_triggering_messages = [ # Renamed variable
        Message(role="user", content="What are your opening hours?"),
        Message(role="assistant", content="We are open from 9 AM to 5 PM, Monday to Friday."),
        Message(role="user", content="Okay, thank you for the information.")
    ]
    non_triggering_conversation = Conversation(messages=non_triggering_messages) # Renamed variable
    print("Non-triggering conversation (No PII) created.")

    # SUBSECTION 2.4: Process Triggering Conversation
    print("\nProcessing the triggering conversation (Contains PII)...")
    triggering_result = guardrail.check(conversation=triggering_conversation) # Renamed variable
    print("Processing complete.")

    # SUBSECTION 2.5: Print Triggering Conversation Result
    # Expected fields in `details` for PIIProtectionGuardrail:
    # 'pii_detected': bool
    # 'pii_types': list of str (e.g., ['EMAIL', 'PHONE', 'ADDRESS', 'FULL_NAME'])
    # 'redaction_suggestions': list of dicts or str (detailing what to redact and how)
    # 'confidence_scores': dict (mapping PII type to confidence)
    print("\n--- Results for Triggering Conversation (Contains PII) ---")
    print(f"Guardrail Triggered (PII Detected): {triggering_result.is_triggered}") # Renamed variable
    # Expected fields in `details` for PIIProtectionGuardrail from its `check` method:
    # 'pii_types': list of str (e.g., ['EMAIL', 'PHONE', 'ADDRESS', 'FULL_NAME'])
    # 'pii_details': list of dicts (specific instances of PII found)
    # 'summary': str (summary of PII findings)
    # 'parsed_llm_output': dict (The full parsed dictionary from the LLM response)
    print(f"Details: {triggering_result.details}") # Print .details directly
    print(f"Errors: {triggering_result.errors}") # Print .errors
    print(f"Metadata: {triggering_result.metadata}") # Print .metadata

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    print("\nProcessing the non-triggering conversation (No PII)...")
    non_triggering_result = guardrail.check(conversation=non_triggering_conversation) # Renamed variable
    print("Processing complete.")

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    print("\n--- Results for Non-Triggering Conversation (No PII) ---")
    print(f"Guardrail Triggered (PII Detected): {non_triggering_result.is_triggered}") # Renamed variable
    print(f"Details: {non_triggering_result.details}") # Print .details directly
    print(f"Errors: {non_triggering_result.errors}") # Print .errors
    print(f"Metadata: {non_triggering_result.metadata}") # Print .metadata

# SECTION 3: Execute main function
if __name__ == "__main__":
    main()