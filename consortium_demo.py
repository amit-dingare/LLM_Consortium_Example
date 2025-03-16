import os
import json
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from llm_consortium import create_consortium

def visualize_consortium_results(result):
    """
    Visualizes the orchestration process with confidence levels and model interactions.
    
    Args:
        result: The raw result from the orchestrator
    """
    # Extract iterations data
    iterations = result.get('iterations', [])
    if not iterations:
        print("No iteration data available for visualization")
        return
    
    # Create a DataFrame for confidence tracking
    confidence_data = []
    for i, iteration in enumerate(iterations):
        for model_name, model_data in iteration.get('responses', {}).items():
            if isinstance(model_data, dict) and 'confidence' in model_data:
                confidence_data.append({
                    'iteration': i+1,
                    'model': model_name,
                    'confidence': float(model_data['confidence'])
                })
    
    if not confidence_data:
        print("No confidence data available")
        return
        
    confidence_df = pd.DataFrame(confidence_data)
    
    # Plot confidence over iterations
    plt.figure(figsize=(10, 6))
    for model in confidence_df['model'].unique():
        model_data = confidence_df[confidence_df['model'] == model]
        plt.plot(model_data['iteration'], model_data['confidence'], 'o-', label=model)
    
    plt.xlabel('Iteration')
    plt.ylabel('Confidence')
    plt.title('Model Confidence Across Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, 1.1)  # Confidence is typically between 0 and 1
    
    # Save the confidence plot
    plt.savefig('confidence_plot.png')
    print("Saved confidence visualization to confidence_plot.png")
    
    # Create a graph visualization of the orchestration
    G = nx.DiGraph()
    
    # Add nodes for models
    model_names = set()
    for iteration in iterations:
        for model_name in iteration.get('responses', {}).keys():
            model_names.add(model_name)
    
    # Add the arbiter
    model_names.add('arbiter')
    
    # Add nodes
    for model in model_names:
        G.add_node(model)
    
    # Add edges for interactions
    for i, iteration in enumerate(iterations):
        # Models submit answers to arbiter
        for model_name in iteration.get('responses', {}).keys():
            G.add_edge(model_name, 'arbiter', iteration=i+1)
        
        # Arbiter provides feedback
        if i < len(iterations) - 1:  # If not the last iteration
            for model_name in iterations[i+1].get('responses', {}).keys():
                G.add_edge('arbiter', model_name, iteration=i+1)
    
    # Create the graph visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in G.nodes if n != 'arbiter'], 
                          node_color='skyblue', 
                          node_size=1500)
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=['arbiter'], 
                          node_color='orange', 
                          node_size=2000)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title('LLM Consortium Orchestration Flow')
    plt.axis('off')
    plt.savefig('orchestration_graph.png')
    print("Saved orchestration graph to orchestration_graph.png")
    
    # If we have a final synthesis, display it
    if 'synthesis' in result and 'synthesis' in result['synthesis']:
        print("\nFinal Synthesis:")
        print(result['synthesis']['synthesis'])

def main():
    # First, let's list available models to find the correct names
    try:
        import llm
        print("Available models:")
        for model in llm.get_models():
            print(f" - {model.model_id}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
    # Configure the consortium with the requested model names
    orchestrator = create_consortium(
        models=[
            "anthropic/claude-3-7-sonnet-latest",
            "gpt-4o",
            "anthropic/claude-3-opus-20240229",
        ],
        confidence_threshold=0.95,
        max_iterations=8,
        min_iterations=2,
        arbiter="gemini-2.0-flash",
        raw=True
    )
    
    # Define the prompt
    prompt = "𝘍𝘰𝘶𝘳 𝘨𝘰𝘭𝘧𝘦𝘳𝘴 𝘯𝘢𝘮𝘦𝘥 𝘔𝘳. 𝘉𝘭𝘢𝘤𝘬, 𝘔𝘳. 𝘞𝘩𝘪𝘵𝘦, 𝘔𝘳. 𝘉𝘳𝘰𝘸𝘯 𝘢𝘯𝘥 𝘔𝘳. 𝘉𝘭𝘶𝘦 𝘸𝘦𝘳𝘦 𝘤𝘰𝘮𝘱𝘦𝘵𝘪𝘯𝘨 𝘪𝘯 𝘢 𝘵𝘰𝘶𝘳𝘯𝘢𝘮𝘦𝘯𝘵. 𝘛𝘩𝘦 𝘤𝘢𝘥𝘥𝘺 𝘥𝘪𝘥𝘯'𝘵 𝘬𝘯𝘰𝘸 𝘵𝘩𝘦𝘪𝘳 𝘯𝘢𝘮𝘦𝘴, " \
            "𝘴𝘰 𝘩𝘦 𝘢𝘴𝘬𝘦𝘥 𝘵𝘩𝘦𝘮. 𝘖𝘯𝘦 𝘰𝘧 𝘵𝘩𝘦𝘮, 𝘔𝘳. 𝘉𝘳𝘰𝘸𝘯, 𝘵𝘰𝘭𝘥 𝘢 𝘭𝘪𝘦. 𝘛𝘩𝘦 1𝘴𝘵 𝘨𝘰𝘭𝘧𝘦𝘳 𝘴𝘢𝘪𝘥 𝘛𝘩𝘦 2𝘯𝘥 𝘎𝘰𝘭𝘧𝘦𝘳 𝘪𝘴 𝘔𝘳. 𝘉𝘭𝘢𝘤𝘬. 𝘛𝘩𝘦 2𝘯𝘥 𝘨𝘰𝘭𝘧𝘦𝘳 𝘴𝘢𝘪𝘥 𝘐 𝘢𝘮 𝘯𝘰𝘵 𝘔𝘳. 𝘉𝘭𝘶𝘦!" \
            "𝘛𝘩𝘦 3𝘳𝘥 𝘨𝘰𝘭𝘧𝘦𝘳 𝘴𝘢𝘪𝘥 𝘔𝘳. 𝘞𝘩𝘪𝘵𝘦? 𝘛𝘩𝘢𝘵'𝘴 𝘵𝘩𝘦 4𝘵𝘩 𝘨𝘰𝘭𝘧𝘦𝘳. 𝘈𝘯𝘥 𝘵𝘩𝘦 4𝘵𝘩 𝘨𝘰𝘭𝘧𝘦𝘳 𝘳𝘦𝘮𝘢𝘪𝘯𝘦𝘥 𝘴𝘪𝘭𝘦𝘯𝘵. 𝘞𝘩𝘪𝘤𝘩 𝘰𝘯𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘨𝘰𝘭𝘧𝘦𝘳𝘴 𝘪𝘴 𝘔𝘳. 𝘉𝘭𝘶𝘦?"
    
    try:
        print(f"Sending prompt: '{prompt}'")
        result = orchestrator.orchestrate(prompt)
        
        # Save raw result for debugging
        with open('consortium_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("Saved raw result to consortium_result.json")

        # Print the final synthesis
        if 'synthesis' in result and 'synthesis' in result['synthesis']:
            print("\nFinal Synthesis:")
            print(result['synthesis']['synthesis'])
        
        # Visualize the orchestration process
        visualize_consortium_results(result)
        
    except Exception as e:
        print(f"Error orchestrating response: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    print("Running the LLM Consortium demo with visualization...")
    main()