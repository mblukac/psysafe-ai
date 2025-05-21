# Specification: Suicide Prevention Guardrail Example Script

## 1. Introduction and Purpose

This document specifies the design and functionality for an example Python script, [`examples/02_suicide_prevention.py`](examples/02_suicide_prevention.py). The script's primary purpose is to demonstrate the basic usage of the `SuicidePreventionGuardrail` available in the `psysafe` library. It will show how to initialize the guardrail, process conversations, and interpret the results.

## 2. Target Audience

This example script is intended for developers and users of the `psysafe` library who want to understand and integrate the `SuicidePreventionGuardrail` into their applications.

## 3. File Location

The Python script will be located at: [`examples/02_suicide_prevention.py`](examples/02_suicide_prevention.py)

## 4. Functional Requirements

The example script must:
1.  Import necessary modules:
    *   [`SuicidePreventionGuardrail`](psysafe/catalog/suicide_prevention/guardrail.py) from `psysafe.catalog.suicide_prevention.guardrail`.
    *   [`Conversation`](psysafe/core/models.py) and [`Message`](psysafe/core/models.py) from `psysafe.core.models`.
2.  Initialize the [`SuicidePreventionGuardrail`](psysafe/catalog/suicide_prevention/guardrail.py) using its default configuration.
3.  Create a sample [`Conversation`](psysafe/core/models.py) that is designed to clearly trigger the guardrail (e.g., containing messages with suicidal ideation).
4.  Create a second sample [`Conversation`](psysafe/core/models.py) that is designed to *not* trigger the guardrail (e.g., containing neutral or positive messages).
5.  Process both conversations using the guardrail's `check()` method.
6.  Print the results of each check, clearly indicating:
    *   Whether the guardrail was triggered.
    *   Any associated details or metadata from the [`CheckResult`](psysafe/core/check.py) object.
7.  Include clear comments throughout the code to explain each step.
8.  Be simple, runnable, and easy for a new user to understand.
9.  Not require any external API keys or complex setup procedures.

## 5. Key Components

*   **[`SuicidePreventionGuardrail`](psysafe/catalog/suicide_prevention/guardrail.py):** The core component responsible for detecting potential suicide risk in conversations.
*   **[`Conversation`](psysafe/core/models.py):** A data structure representing a sequence of messages.
*   **[`Message`](psysafe/core/models.py):** A data structure representing a single message within a conversation, including its content and role (e.g., "user").
*   **[`CheckResult`](psysafe/core/check.py):** An object returned by the guardrail's `check()` method, containing the outcome of the safety check.

## 6. Pseudocode

```pseudocode
# File: examples/02_suicide_prevention.py

# SECTION 1: Imports
# Import SuicidePreventionGuardrail from psysafe.catalog.suicide_prevention.guardrail
# Import Conversation, Message from psysafe.core.models
# (Optionally, import Role if it's an enum, otherwise use string literals like "user")

# SECTION 2: Define a main function or execution block
# def main():

    # SUBSECTION 2.1: Initialize Guardrail
    # Print "Initializing SuicidePreventionGuardrail..."
    # guardrail = SuicidePreventionGuardrail()
    # Print "Guardrail initialized."
    # TDD Anchor: test_guardrail_initialization_succeeds()

    # SUBSECTION 2.2: Create Triggering Conversation
    # Print "\nCreating a sample conversation designed to trigger the guardrail..."
    # triggering_messages = [
    #     Message(role="user", content="I feel so hopeless and alone."),
    #     Message(role="user", content="I'm thinking about ending my life, there's no other way out.")
    # ]
    # triggering_conversation = Conversation(messages=triggering_messages)
    # Print "Triggering conversation created."
    # TDD Anchor: test_triggering_conversation_structure_is_correct()

    # SUBSECTION 2.3: Create Non-Triggering Conversation
    # Print "\nCreating a sample conversation designed NOT to trigger the guardrail..."
    # non_triggering_messages = [
    #     Message(role="user", content="I had a great day today!"),
    #     Message(role="user", content="Planning a picnic for the weekend.")
    # ]
    # non_triggering_conversation = Conversation(messages=non_triggering_messages)
    # Print "Non-triggering conversation created."
    # TDD Anchor: test_non_triggering_conversation_structure_is_correct()

    # SUBSECTION 2.4: Process Triggering Conversation
    # Print "\nProcessing the triggering conversation..."
    # # The check method is expected to return a CheckResult object
    # triggering_result = guardrail.check(conversation=triggering_conversation)
    # Print "Processing complete."
    # TDD Anchor: test_guardrail_detects_triggering_conversation_correctly()

    # SUBSECTION 2.5: Print Triggering Conversation Result
    # Print "\n--- Results for Triggering Conversation ---"
    # Print(f"Guardrail Triggered: {triggering_result.is_triggered}") # Assuming CheckResult has is_triggered
    # Print(f"Risk Score: {triggering_result.metadata.get('risk_score', 'N/A')}") # Assuming metadata structure
    # Print(f"Details: {triggering_result.metadata}") # Or specific fields from metadata
    # TDD Anchor: test_triggering_conversation_output_contains_expected_info()

    # SUBSECTION 2.6: Process Non-Triggering Conversation
    # Print "\nProcessing the non-triggering conversation..."
    # non_triggering_result = guardrail.check(conversation=non_triggering_conversation)
    # Print "Processing complete."
    # TDD Anchor: test_guardrail_ignores_non_triggering_conversation_correctly()

    # SUBSECTION 2.7: Print Non-Triggering Conversation Result
    # Print "\n--- Results for Non-Triggering Conversation ---"
    # Print(f"Guardrail Triggered: {non_triggering_result.is_triggered}")
    # Print(f"Risk Score: {non_triggering_result.metadata.get('risk_score', 'N/A')}")
    # Print(f"Details: {non_triggering_result.metadata}")
    # TDD Anchor: test_non_triggering_conversation_output_contains_expected_info()

# SECTION 3: Execute main function
# if __name__ == "__main__":
#     main()
```

## 7. Expected Output (Illustrative)

The script's output should resemble the following:

```
Initializing SuicidePreventionGuardrail...
Guardrail initialized.

Creating a sample conversation designed to trigger the guardrail...
Triggering conversation created.

Creating a sample conversation designed NOT to trigger the guardrail...
Non-triggering conversation created.

Processing the triggering conversation...
Processing complete.

--- Results for Triggering Conversation ---
Guardrail Triggered: True
Risk Score: 0.95 
Details: {'risk_score': 0.95, 'reason': 'Detected strong suicidal ideation.', 'triggered_messages': [1]} 

Processing the non-triggering conversation...
Processing complete.

--- Results for Non-Triggering Conversation ---
Guardrail Triggered: False
Risk Score: 0.05
Details: {'risk_score': 0.05, 'reason': 'No significant suicide risk detected.', 'triggered_messages': []}
```
*(Note: The exact structure and content of `metadata` might vary based on the `SuicidePreventionGuardrail` implementation. The example shows plausible fields.)*

## 8. TDD Anchors

The following TDD anchors should be used to guide test development:

1.  `test_guardrail_initialization_succeeds()`: Verifies that [`SuicidePreventionGuardrail`](psysafe/catalog/suicide_prevention/guardrail.py) can be instantiated without errors using default settings.
2.  `test_triggering_conversation_structure_is_correct()`: Ensures the [`Conversation`](psysafe/core/models.py) object intended to trigger the guardrail is created correctly with appropriate [`Message`](psysafe/core/models.py) objects.
3.  `test_non_triggering_conversation_structure_is_correct()`: Ensures the [`Conversation`](psysafe/core/models.py) object intended *not* to trigger the guardrail is created correctly.
4.  `test_guardrail_detects_triggering_conversation_correctly()`: Confirms that the `check()` method returns a [`CheckResult`](psysafe/core/check.py) indicating `is_triggered = True` for the triggering conversation.
5.  `test_triggering_conversation_output_contains_expected_info()`: Checks that the script's printout for the triggering conversation includes the trigger status and relevant metadata.
6.  `test_guardrail_ignores_non_triggering_conversation_correctly()`: Confirms that the `check()` method returns a [`CheckResult`](psysafe/core/check.py) indicating `is_triggered = False` for the non-triggering conversation.
7.  `test_non_triggering_conversation_output_contains_expected_info()`: Checks that the script's printout for the non-triggering conversation includes the trigger status (False) and relevant metadata.

This specification provides a clear plan for developing the example script.