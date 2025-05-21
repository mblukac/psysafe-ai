# File: examples/03_complaints_handling.py

# SECTION 1: Imports
from dotenv import load_dotenv
from pathlib import Path

from psysafe.catalog.complaints_handling.guardrail import ComplaintsHandlingGuardrail
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
    Main function to demonstrate the ComplaintsHandlingGuardrail.
    """
    # SUBSECTION 2.1: Initialize Guardrail
    print("Initializing ComplaintsHandlingGuardrail...")
    guardrail = ComplaintsHandlingGuardrail() # Add parameters if any, e.g., sensitivity
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

    # SUBSECTION 2.2: Create Triggering Conversation (User Complaint)
    print("\nCreating a sample conversation designed to trigger the guardrail (User Complaint)...")
    triggering_messages_complaint = [
        Message(role="user", content="I'm very unhappy with the service I received. The product broke after one day!"),
        Message(role="assistant", content="I'm sorry to hear that. Can you tell me more about the issue?"),
        Message(role="user", content="It was the new SuperWidget X1000. It just stopped working and I want a full refund immediately. This is unacceptable!")
    ]
    triggering_conversation_complaint = Conversation(messages=triggering_messages_complaint)
    print("Triggering conversation (User Complaint) created.")

    # SUBSECTION 2.3: Create Non-Triggering Conversation (General Inquiry)
    print("\nCreating a sample conversation designed NOT to trigger the guardrail (General Inquiry)...")
    non_triggering_messages_inquiry = [
        Message(role="user", content="Hello, I'd like to know more about your products."),
        Message(role="assistant", content="Certainly! What product are you interested in?"),
        Message(role="user", content="I was looking at the specifications for the SuperWidget X1000. Can you tell me its battery life?")
    ]
    non_triggering_conversation_inquiry = Conversation(messages=non_triggering_messages_inquiry)
    print("Non-triggering conversation (General Inquiry) created.")

    # SUBSECTION 2.4: Process Triggering Conversation
    print("\nProcessing the triggering conversation (User Complaint)...")
    triggering_result_complaint = guardrail.check(conversation=triggering_conversation_complaint)
    print("Processing complete.")

    # SUBSECTION 2.5: Print Triggering Conversation Result
    # Expected fields in `details` for ComplaintsHandlingGuardrail:
    # 'is_complaint': bool
    # 'category': str (e.g., 'Product Quality', 'Service Issue')
    # 'urgency': str (e.g., 'High', 'Medium', 'Low')
    # 'summary': str
    # 'suggested_action': str
    print("\n--- Results for Triggering Conversation (User Complaint) ---")
    print(f"Guardrail Triggered (Is Complaint): {triggering_result_complaint.is_triggered}")
    print(f"Category: {triggering_result_complaint.details.get('category', 'N/A')}")
    print(f"Urgency: {triggering_result_complaint.details.get('urgency', 'N/A')}")
    print(f"Summary: {triggering_result_complaint.details.get('summary', 'N/A')}")
    print(f"Suggested Action: {triggering_result_complaint.details.get('suggested_action', 'N/A')}")
    print(f"Full Details: {triggering_result_complaint.details}")
    print(f"Errors: {triggering_result_complaint.errors}")
    print(f"Metadata: {triggering_result_complaint.metadata}")

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    print("\nProcessing the non-triggering conversation (General Inquiry)...")
    non_triggering_result_inquiry = guardrail.check(conversation=non_triggering_conversation_inquiry)
    print("Processing complete.")

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    print("\n--- Results for Non-Triggering Conversation (General Inquiry) ---")
    print(f"Guardrail Triggered (Is Complaint): {non_triggering_result_inquiry.is_triggered}")
    print(f"Category: {non_triggering_result_inquiry.details.get('category', 'N/A')}")
    print(f"Urgency: {non_triggering_result_inquiry.details.get('urgency', 'N/A')}")
    print(f"Summary: {non_triggering_result_inquiry.details.get('summary', 'N/A')}")
    print(f"Suggested Action: {non_triggering_result_inquiry.details.get('suggested_action', 'N/A')}")
    print(f"Full Details: {non_triggering_result_inquiry.details}")
    print(f"Errors: {non_triggering_result_inquiry.errors}")
    print(f"Metadata: {non_triggering_result_inquiry.metadata}")

# SECTION 3: Execute main function
if __name__ == "__main__":
    main()