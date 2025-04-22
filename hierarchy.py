import networkx as nx

# SemEval 2021 and 2024 use synomymous technique names
LABEL_MAP = {
    "Appeal to authority": "Appeal_to_Authority",
    "Appeal to fear/prejudice": "Appeal_to_Fear-Prejudice",
    "Bandwagon": "Appeal_to_Popularity",
    "Glittering generalities (Virtue)": "Appeal_to_Values",
    "Causal Oversimplification": "Causal_Oversimplification",
    "Thought-terminating cliché": "Conversation_Killer",
    "Doubt": "Doubt",
    "Exaggeration/Minimisation": "Exaggeration-Minimisation",
    "Black-and-white Fallacy/Dictatorship": "False_Dilemma-No_Choice",
    "Flag-waving": "Flag_Waving",
    "Reductio ad hitlerum": "Guilt_by_Association",
    "Loaded Language": "Loaded_Language",
    "Name calling/Labeling": "Name_Calling-Labeling",
    "Obfuscation, Intentional vagueness, Confusion": "Obfuscation-Vagueness-Confusion",
    "Smears": "Questioning_the_Reputation",
    "Presenting Irrelevant Data (Red Herring)": "Red_Herring",
    "Repetition": "Repetition",
    "Slogans": "Slogans",
    "Misrepresentation of Someone's Position (Straw Man)": "Straw_Man",
    "Whataboutism": "Whataboutism",
}


def create_full_hierarchy() -> nx.DiGraph:
    """Create the full label hierarchy DAG defined by SemEval 2024 Task 4 and extended
    with the new labels based on the technique definitions and the label taxonomy defined
    in the Slavic NLP 2025 shared task."""
    G = nx.DiGraph()
    G.add_edge("Persuasion", "Logos")
    G.add_edge("Logos", "Repetition")
    G.add_edge("Logos", "Obfuscation-Vagueness-Confusion")
    G.add_edge("Logos", "Reasoning")
    G.add_edge("Logos", "Justification")
    G.add_edge("Justification", "Slogans")
    G.add_edge("Justification", "Appeal_to_Popularity")
    G.add_edge("Justification", "Appeal_to_Authority")
    G.add_edge("Justification", "Flag_Waving")
    G.add_edge("Justification", "Appeal_to_Fear-Prejudice")
    G.add_edge("Reasoning", "Simplification")
    G.add_edge("Simplification", "Causal_Oversimplification")
    G.add_edge("Simplification", "False_Dilemma-No_Choice")
    G.add_edge("Simplification", "Conversation_Killer")
    G.add_edge("Simplification", "Consequential_Oversimplification")
    G.add_edge("Simplification", "False_Equivalence")  # new label
    G.add_edge("Reasoning", "Distraction")
    G.add_edge("Distraction", "Straw_Man")
    G.add_edge("Distraction", "Red_Herring")
    G.add_edge("Distraction", "Whataboutism")
    G.add_edge("Distraction", "Appeal_to_Pity")  # "Appeal to Emotion" in SemEval 2024
    G.add_edge("Persuasion", "Ethos")
    G.add_edge("Ethos", "Appeal_to_Authority")
    G.add_edge("Ethos", "Appeal_to_Values")
    G.add_edge("Ethos", "Appeal_to_Popularity")
    G.add_edge("Ethos", "Ad Hominem")
    G.add_edge("Ad Hominem", "Doubt")
    G.add_edge("Ad Hominem", "Name_Calling-Labeling")
    G.add_edge("Ad Hominem", "Questioning_the_Reputation")
    G.add_edge("Ad Hominem", "Guilt_by_Association")
    G.add_edge("Ad Hominem", "Appeal_to_Hypocrisy")
    G.add_edge("Ad Hominem", "Whataboutism")
    G.add_edge("Persuasion", "Pathos")
    G.add_edge("Pathos", "Exaggeration-Minimisation")
    G.add_edge("Pathos", "Loaded_Language")
    G.add_edge("Pathos", "Appeal_to_Fear-Prejudice")
    G.add_edge("Pathos", "Flag_Waving")
    G.add_edge("Pathos", "Appeal_to_Pity")  # "Appeal to Emotion" in SemEval 2024
    G.add_edge("Persuasion", "Appeal_to_Time")  # "Kairos", the 4th mode of persuation
    return G


def create_taxonomy() -> nx.DiGraph:
    """Create the label taxonomy tree defined in the Slavic NLP 2025 shared task."""
    G = nx.DiGraph()
    G.add_edge("Persuasion", "Attack_on_Reputation")  # "Ad Hominem" in SemEval 2024
    G.add_edge("Attack_on_Reputation", "Name_Calling-Labeling")
    G.add_edge("Attack_on_Reputation", "Guilt_by_Association")
    G.add_edge("Attack_on_Reputation", "Doubt")
    G.add_edge("Attack_on_Reputation", "Appeal_to_Hypocrisy")
    G.add_edge("Attack_on_Reputation", "Questioning_the_Reputation")
    G.add_edge("Persuasion", "Justification")
    G.add_edge("Justification", "Flag_Waving")
    G.add_edge("Justification", "Appeal_to_Authority")
    G.add_edge("Justification", "Appeal_to_Popularity")
    G.add_edge("Justification", "Appeal_to_Fear-Prejudice")
    G.add_edge("Justification", "Appeal_to_Values")
    G.add_edge("Persuasion", "Distraction")
    G.add_edge("Distraction", "Straw_Man")
    G.add_edge("Distraction", "Whataboutism")
    G.add_edge("Distraction", "Red_Herring")
    G.add_edge("Distraction", "Appeal_to_Pity")
    G.add_edge("Persuasion", "Simplification")
    G.add_edge("Simplification", "Causal_Oversimplification")
    G.add_edge("Simplification", "False_Dilemma-No_Choice")
    G.add_edge("Simplification", "Consequential_Oversimplification")
    G.add_edge("Simplification", "False_Equivalence")
    G.add_edge("Persuasion", "Call")  # does not exist in SemEval 2024
    G.add_edge("Call", "Slogans")  # belongs to Logos/Justification in SemEval 2024
    G.add_edge("Call", "Conversation_Killer")  # belongs to Logos/Simplification in SemEval 2024
    G.add_edge("Call", "Appeal_to_Time")  # "Kairos", the 4th mode of persuation
    G.add_edge("Persuasion", "Manipulative_Wording")  # called Pathos in SemEval 2024
    G.add_edge("Manipulative_Wording", "Loaded_Language")
    G.add_edge("Manipulative_Wording", "Obfuscation-Vagueness-Confusion")
    G.add_edge("Manipulative_Wording", "Exaggeration-Minimisation")
    G.add_edge("Manipulative_Wording", "Repetition")  # belongs to Logos in SemEval 2024
    return G
