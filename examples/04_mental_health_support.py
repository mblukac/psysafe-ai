# File: examples/04_mental_health_support.py

# SECTION 1: Imports
from dotenv import load_dotenv
from pathlib import Path

from psysafe.catalog.mental_health_support.guardrail import MentalHealthSupportGuardrail
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
    Main function to demonstrate the MentalHealthSupportGuardrail.
    """
    # SUBSECTION 2.1: Initialize Guardrail
    print("Initializing MentalHealthSupportGuardrail...")
    # Consider if this guardrail has parameters like sensitivity
    guardrail = MentalHealthSupportGuardrail()
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

    # SUBSECTION 2.2: Create Triggering Conversation (User Expressing Need for Support)
    print("\nCreating a sample conversation designed to trigger the guardrail (Need for Support)...")
    triggering_messages_support = [
        Message(role="user", content="I've been feeling really down and anxious lately. It's hard to cope."),
        Message(role="assistant", content="I understand that can be tough. How long have you been feeling this way?"),
        Message(role="user", content="For a few weeks now. I just don't know who to turn to or what to do.")
    ]
    triggering_conversation_support = Conversation(messages=triggering_messages_support)
    print("Triggering conversation (Need for Support) created.")

    # SUBSECTION 2.3: Create Non-Triggering Conversation (General Chat)
    print("\nCreating a sample conversation designed NOT to trigger the guardrail (General Chat)...")
    non_triggering_messages_chat = [
        Message(role="user", content="What's the weather like today?"),
        Message(role="assistant", content="It's sunny with a high of 25 degrees Celsius."),
        Message(role="user", content="Great, thanks!")
    ]
    non_triggering_conversation_chat = Conversation(messages=non_triggering_messages_chat)
    print("Non-triggering conversation (General Chat) created.")

    # SUBSECTION 2.4: Process Triggering Conversation
    print("\nProcessing the triggering conversation (Need for Support)...")
    triggering_result_support = guardrail.check(conversation=triggering_conversation_support)
    print("Processing complete.")

    # SUBSECTION 2.5: Print Triggering Conversation Result
    # Expected fields in `details` for MentalHealthSupportGuardrail:
    # 'needs_support': bool
    # 'support_category': str (e.g., 'Anxiety', 'Depression', 'Stress')
    # 'severity_level': str (e.g., 'Mild', 'Moderate', 'Severe')
    # 'suggested_action': str (e.g., 'Recommend professional help', 'Offer coping strategies')
    # 'crisis_level': str (e.g., 'None', 'Low', 'Medium', 'High', 'Critical')
    print("\n--- Results for Triggering Conversation (Need for Support) ---")
    print(f"Guardrail Triggered (Needs Support): {triggering_result_support.is_triggered}")
    print(f"Support Category: {triggering_result_support.details.get('support_category', 'N/A')}")
    print(f"Severity Level: {triggering_result_support.details.get('severity_level', 'N/A')}")
    print(f"Suggested Action: {triggering_result_support.details.get('suggested_action', 'N/A')}")
    print(f"Crisis Level: {triggering_result_support.details.get('crisis_level', 'N/A')}")
    print(f"Full Details: {triggering_result_support.details}")
    print(f"Errors: {triggering_result_support.errors}")
    print(f"Metadata: {triggering_result_support.metadata}")

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    print("\nProcessing the non-triggering conversation (General Chat)...")
    non_triggering_result_chat = guardrail.check(conversation=non_triggering_conversation_chat)
    print("Processing complete.")

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    print("\n--- Results for Non-Triggering Conversation (General Chat) ---")
    print(f"Guardrail Triggered (Needs Support): {non_triggering_result_chat.is_triggered}")
    print(f"Support Category: {non_triggering_result_chat.details.get('support_category', 'N/A')}")
    print(f"Severity Level: {non_triggering_result_chat.details.get('severity_level', 'N/A')}")
    print(f"Suggested Action: {non_triggering_result_chat.details.get('suggested_action', 'N/A')}")
    print(f"Crisis Level: {non_triggering_result_chat.details.get('crisis_level', 'N/A')}")
    print(f"Full Details: {non_triggering_result_chat.details}")
    print(f"Errors: {non_triggering_result_chat.errors}")
    print(f"Metadata: {non_triggering_result_chat.metadata}")

# SECTION 3: Execute main function
if __name__ == "__main__":
    main()