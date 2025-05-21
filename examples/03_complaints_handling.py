# File: examples/03_complaints_handling.py

# SECTION 1: Imports
from utils.llm_utils import load_environment # Changed import
from psysafe.catalog.complaints_handling.guardrail import ComplaintsHandlingGuardrail
from psysafe.core.models import Conversation, Message
from psysafe.drivers.openai import OpenAIChatDriver

load_environment() # Added environment loading call

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
    triggering_messages = [ # Renamed variable
        Message(role="user", content="I'm very unhappy with the service I received. The product broke after one day!"),
        Message(role="assistant", content="I'm sorry to hear that. Can you tell me more about the issue?"),
        Message(role="user", content="It was the new SuperWidget X1000. It just stopped working and I want a full refund immediately. This is unacceptable!")
    ]
    triggering_conversation = Conversation(messages=triggering_messages) # Renamed variable
    print("Triggering conversation (User Complaint) created.")

    # SUBSECTION 2.3: Create Non-Triggering Conversation (General Inquiry)
    print("\nCreating a sample conversation designed NOT to trigger the guardrail (General Inquiry)...")
    non_triggering_messages = [ # Renamed variable
        Message(role="user", content="Hello, I'd like to know more about your products."),
        Message(role="assistant", content="Certainly! What product are you interested in?"),
        Message(role="user", content="I was looking at the specifications for the SuperWidget X1000. Can you tell me its battery life?")
    ]
    non_triggering_conversation = Conversation(messages=non_triggering_messages) # Renamed variable
    print("Non-triggering conversation (General Inquiry) created.")

    # SUBSECTION 2.4: Process Triggering Conversation
    print("\nProcessing the triggering conversation (User Complaint)...")
    triggering_result = guardrail.check(conversation=triggering_conversation) # Renamed variable
    print("Processing complete.")

    # SUBSECTION 2.5: Print Triggering Conversation Result
    # Expected fields in `details` for ComplaintsHandlingGuardrail:
    # 'is_complaint': bool
    # 'category': str (e.g., 'Product Quality', 'Service Issue')
    # 'urgency': str (e.g., 'High', 'Medium', 'Low')
    # 'summary': str
    # 'suggested_action': str
    print("\n--- Results for Triggering Conversation (User Complaint) ---")
    print(f"Guardrail Triggered (Is Complaint): {triggering_result.is_triggered}") # Renamed variable
    # The `details` field from ComplaintsHandlingGuardrail.check() is expected to contain:
    # 'category': str (e.g., 'Product Quality', 'Service Issue', 'Billing')
    # 'summary': str (A brief summary of the complaint)
    # 'escalation_needed': bool (Whether the LLM determined escalation is needed)
    # 'parsed_llm_output': dict (The full parsed dictionary from the LLM response)
    print(f"Details: {triggering_result.details}") # Print .details directly
    print(f"Errors: {triggering_result.errors}") # Print .errors
    print(f"Metadata: {triggering_result.metadata}") # Print .metadata

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    print("\nProcessing the non-triggering conversation (General Inquiry)...")
    non_triggering_result = guardrail.check(conversation=non_triggering_conversation) # Renamed variable
    print("Processing complete.")

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    print("\n--- Results for Non-Triggering Conversation (General Inquiry) ---")
    print(f"Guardrail Triggered (Is Complaint): {non_triggering_result.is_triggered}") # Renamed variable
    print(f"Details: {non_triggering_result.details}") # Print .details directly
    print(f"Errors: {non_triggering_result.errors}") # Print .errors
    print(f"Metadata: {non_triggering_result.metadata}") # Print .metadata

# SECTION 3: Execute main function
if __name__ == "__main__":
    main()