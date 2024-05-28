import wikipedia

# Function to fetch summary of a Wikipedia page
def get_wikipedia_summary(topic):
    try:
        page = wikipedia.page(topic)
        return page.summary
    except wikipedia.exceptions.PageError as e:
        suggestions = wikipedia.search(topic)
        if suggestions:
            return f"Page not found. Did you mean: {', '.join(suggestions)}?"
        else:
            return "Page not found. No suggestions available."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous topic. Did you mean: {', '.join(e.options)}?"

# Example usage
topic = "Theory of relativity"
summary = get_wikipedia_summary(topic)
print(f"Summary of {topic}:\n{summary}")
