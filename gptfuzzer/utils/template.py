QUESTION_PLACEHOLDER = '[INSERT QUESTION HERE]'

def synthesis_message(prompt, question):
	if QUESTION_PLACEHOLDER not in prompt:
		return None

	return prompt.replace(QUESTION_PLACEHOLDER, question)
