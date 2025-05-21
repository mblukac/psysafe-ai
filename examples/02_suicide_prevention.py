# File: examples/02_suicide_prevention.py

# SECTION 1: Imports
from utils.llm_utils import load_environment
from psysafe.catalog.suicide_prevention.guardrail import SuicidePreventionGuardrail, Sensitivity
from psysafe.core.models import Conversation, Message
from psysafe.drivers.openai import OpenAIChatDriver # Import the driver
# Role enum is not strictly necessary if using string literals, but good for clarity if available
# from psysafe.core.models import Role # Assuming Role might be an enum e.g., Role.USER

load_environment()

def main():
    """
    Main function to demonstrate the SuicidePreventionGuardrail.
    """
    # SUBSECTION 2.1: Initialize Guardrail
    print("Initializing SuicidePreventionGuardrail...")
    guardrail = SuicidePreventionGuardrail(sensitivity=Sensitivity.MEDIUM)
    print("Guardrail initialized.")

    # SUBSECTION 2.1.1: Initialize and Bind LLM Driver
    print("Initializing LLM driver (OpenAIDriver)...")
    # Assumes OPENAI_API_KEY is set in the environment
    try:
        driver = OpenAIChatDriver(model="gpt-4.1-mini-2025-04-14")
        print("LLM driver initialized.")
        print("Binding driver to guardrail...")
        guardrail.bind(driver)
        print("Driver bound to guardrail.")
    except Exception as e:
        print(f"Error initializing or binding LLM driver: {e}")
        print("The script will likely fail at the check step without a functional driver.")
        # Optionally, exit here if a driver is critical for any further steps
        # return

    # SUBSECTION 2.2: Create Triggering Conversation
    print("\nCreating a sample conversation designed to trigger the guardrail...")
    triggering_messages = [
        Message(role="user", content="I feel so hopeless and alone."),
        Message(role="user", content="I'm thinking about ending my life, there's no other way out.")
    ]
    triggering_conversation = Conversation(messages=triggering_messages)
    print("Triggering conversation created.")

    # SUBSECTION 2.3: Create Non-Triggering Conversation
    print("\nCreating a sample conversation designed NOT to trigger the guardrail...")
    non_triggering_messages = [
        Message(role="user", content="I had a great day today!"),
        Message(role="user", content="Planning a picnic for the weekend.")
    ]
    non_triggering_conversation = Conversation(messages=non_triggering_messages)
    print("Non-triggering conversation created.")

    # SUBSECTION 2.4: Process Triggering Conversation
    print("\nProcessing the triggering conversation...")
    # The check method is expected to return a CheckResult object
    triggering_result = guardrail.check(conversation=triggering_conversation)
    print("Processing complete.")

    # SUBSECTION 2.5: Print Triggering Conversation Result
    print("\n--- Results for Triggering Conversation ---")
    print(f"Guardrail Triggered: {triggering_result.is_triggered}")
    # Accessing metadata safely, providing 'N/A' if key is missing or metadata is None
    # The risk_score is now a direct attribute of CheckOutput
    risk_score_triggering = triggering_result.risk_score if triggering_result.risk_score is not None else "N/A"
    
    print(f"Risk Score: {risk_score_triggering}")
    print(f"Details: {triggering_result.details}") # Print .details
    print(f"Errors: {triggering_result.errors}") # Print .errors
    print(f"Metadata: {triggering_result.metadata}") # Print .metadata

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    print("\nProcessing the non-triggering conversation...")
    non_triggering_result = guardrail.check(conversation=non_triggering_conversation)
    print("Processing complete.")

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    print("\n--- Results for Non-Triggering Conversation ---")
    print(f"Guardrail Triggered: {non_triggering_result.is_triggered}")
    # Accessing metadata safely
    risk_score_non_triggering = non_triggering_result.risk_score if non_triggering_result.risk_score is not None else "N/A"

    print(f"Risk Score: {risk_score_non_triggering}")
    print(f"Details: {non_triggering_result.details}") # Print .details
    print(f"Errors: {non_triggering_result.errors}") # Print .errors
    print(f"Metadata: {non_triggering_result.metadata}") # Print .metadata

# SECTION 3: Execute main function
if __name__ == "__main__":
    main()