import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def process_journey_file(filepath):
    output = {}
    
    # Read the Excel file containing journey data
    df = pd.read_csv(filepath)
    df['event'] = df['event'].astype(str).str.strip()
    df = df.sort_values(by=['pid', 'step_order', 'ts'])

    # Distinct PIDs
    distinct_pids = df['pid'].unique()
    output['num_distinct_pids'] = len(distinct_pids)

    # Event transitions
    transitions = defaultdict(int)
    for _, group in df.groupby('pid'):
        events = group.sort_values('step_order')['event'].tolist()
        for i in range(len(events) - 1):
            transitions[(events[i], events[i + 1])] += 1

    G = nx.DiGraph()
    for (src, dst), weight in transitions.items():
        G.add_edge(str(src), str(dst), weight=weight)

    G = nx.relabel_nodes(G, lambda x: str(x))
    node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=2)
    model = node2vec.fit(window=5, min_count=1)
    event_embeddings = {event: model.wv[event] for event in G.nodes()}

    customer_vectors = {}
    for pid, group in df.groupby('pid'):
        events = group.sort_values('step_order')['event'].tolist()
        vectors = [event_embeddings[str(e)] for e in events if str(e) in event_embeddings]
        if vectors:
            customer_vectors[pid] = np.mean(vectors, axis=0)

    if not customer_vectors:
        raise ValueError("No customer vectors found.")

    X = np.array(list(customer_vectors.values()))
    customer_ids = list(customer_vectors.keys())

    X_reduce = PCA(n_components=10).fit_transform(X)
    kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_reduce)

    cluster_df = pd.DataFrame({
        'customer_id': customer_ids,
        'cluster': labels
    })

    total_customers = df['pid'].nunique()
    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    cluster_percentages = (cluster_counts / total_customers * 100).round(2)

    cluster_summary = pd.DataFrame({
        'customer_count': cluster_counts,
        'percentage_of_total': cluster_percentages
    }).reset_index().rename(columns={'index': 'cluster'})

    output['cluster_summary'] = cluster_summary.to_dict(orient='records')

    df = df.merge(cluster_df, left_on='pid', right_on='customer_id', how='left')

    # Transitions per cluster
    cluster_transitions = {}
    for cluster_label, cluster_group in df.groupby('cluster'):
        transitions = defaultdict(int)
        for _, group in cluster_group.groupby('pid'):
            events = group.sort_values('ts')['event'].tolist()
            for i in range(len(events) - 1):
                transitions[(events[i], events[i + 1])] += 1
        cluster_transitions[int(cluster_label)] = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]

    output['top_transitions_per_cluster'] = {
        k: [f"{src} → {dst}: {cnt}" for (src, dst), cnt in v] 
        for k, v in cluster_transitions.items()
    }

    # Successful customers
    successful = df[df['event'] == 'Payment Confirmed']
    output['success_count_per_cluster'] = successful['cluster'].value_counts().to_dict()

    # Path stats (without filtering for goal)
    cluster_paths = {}
    for cluster_id, cluster_group in df.groupby('cluster'):
        paths = []
        for _, group in cluster_group.groupby('pid'):
            events = group.sort_values('step_order')['event'].tolist()
            paths.append(events)
        cluster_paths[cluster_id] = paths

    cluster_path_stats = {}
    for cluster_id, paths in cluster_paths.items():
        path_strings = [' → '.join(path) for path in paths]
        path_counts = Counter(path_strings)
        filtered_paths = [(p, c) for p, c in path_counts.items() if c > 10]
        cluster_path_stats[cluster_id] = filtered_paths[:5]

    output['top_paths_no_goal'] = {
        int(k): [f"{p} — {c} customers" for p, c in v] for k, v in cluster_path_stats.items()
    }

    # Path to Payment Confirmed
    goal_event = 'Payment Confirmed'
    customers_with_goal = df[df['event'] == goal_event]['pid'].unique()
    df_goal = df[df['pid'].isin(customers_with_goal)]

    cluster_paths = {}
    for cluster_id, cluster_group in df_goal.groupby('cluster'):
        paths = []
        for _, group in cluster_group.groupby('pid'):
            events = group.sort_values('step_order')['event'].tolist()
            if goal_event in events:
                path = events[:events.index(goal_event) + 1]
                paths.append(path)
        cluster_paths[cluster_id] = paths

    cluster_path_stats = {}
    for cluster_id, paths in cluster_paths.items():
        path_strings = [' → '.join(path) for path in paths]
        path_counts = Counter(path_strings)
        filtered_paths = [(p, c) for p, c in path_counts.items() if c > 10]
        cluster_path_stats[cluster_id] = filtered_paths[:5]

    output['top_paths_to_payment_confirmed'] = {
        int(k): [f"{p} — {c} customers" for p, c in v] for k, v in cluster_path_stats.items()
    }

    # Path to Payment Submitted
    goal_event = 'Payment Submitted'
    customers_with_goal = df[df['event'] == goal_event]['pid'].unique()
    df_goal = df[df['pid'].isin(customers_with_goal)]

    cluster_paths = {}
    for cluster_id, cluster_group in df_goal.groupby('cluster'):
        paths = []
        for _, group in cluster_group.groupby('pid'):
            events = group.sort_values('step_order')['event'].tolist()
            if goal_event in events:
                paths.append(events)
        cluster_paths[cluster_id] = paths

    cluster_path_stats = {}
    for cluster_id, paths in cluster_paths.items():
        path_strings = [' → '.join(path) for path in paths]
        path_counts = Counter(path_strings)
        filtered_paths = [(p, c) for p, c in path_counts.items() if c > 10]
        cluster_path_stats[cluster_id] = filtered_paths[:5]

    output['top_paths_to_payment_submitted'] = {
        int(k): [f"{p} — {c} customers" for p, c in v] for k, v in cluster_path_stats.items()
    }

    # Last events taken by customers analysis
    df_sorted = df.sort_values(by=['pid', 'ts'])
    last_event_per_user = df_sorted.groupby('pid').tail(1)[['pid', 'event']]
    event_counts = last_event_per_user['event'].value_counts().reset_index()
    event_counts.columns = ['event', 'users_ended_at_event']
    total_users = last_event_per_user['pid'].nunique()
    event_counts['percentage'] = (event_counts['users_ended_at_event'] / total_users * 100).round(2)
    # After calculating event_counts and percentages
    top_20_events = event_counts.head(20)
    output['last_event_summary'] = top_20_events.to_dict(orient='records')

    print(output)
    
    def generate_summary(output):
        lines = []
        lines.append(f"Number of distinct customers: {output['num_distinct_pids']}\n")

        lines.append("Cluster Summary:")
        for cluster in output['cluster_summary']:
            lines.append(f"- Cluster {cluster['cluster']}: {cluster['customer_count']} customers ({cluster['percentage_of_total']}%)")
        lines.append("")

        lines.append("Top Transitions per Cluster:")
        for cluster_id, transitions in output['top_transitions_per_cluster'].items():
            lines.append(f"Cluster {cluster_id}:")
            for t in transitions:
                lines.append(f"  - {t}")
            lines.append("")

        lines.append("Success Count per Cluster:")
        for cluster_id, count in output['success_count_per_cluster'].items():
            lines.append(f"- Cluster {cluster_id}: {count} successful payments")
        lines.append("")

        def add_paths_section(title, paths_dict):
            lines.append(title + ":")
            for cluster_id, paths in paths_dict.items():
                lines.append(f"Cluster {cluster_id}:")
                if paths:
                    for p in paths:
                        lines.append(f"  - {p}")
                else:
                    lines.append("  - None")
                lines.append("")

        add_paths_section("Top Paths Without Goal", output['top_paths_no_goal'])
        add_paths_section("Top Paths to Payment Confirmed", output['top_paths_to_payment_confirmed'])
        add_paths_section("Top Paths to Payment Submitted", output['top_paths_to_payment_submitted'])

        # Add last event summary section
        lines.append("Last Event Summary:")
        if output.get('last_event_summary'):
            for rec in output['last_event_summary']:
                lines.append(f"- Event '{rec['event']}': {rec['users_ended_at_event']} users ended here ({rec['percentage']}%)")
        else:
            lines.append("- No data available")
        lines.append("")

        return "\n".join(lines)
    
    summary_text = generate_summary(output)
    print(summary_text)  # or save/send this summary to the LLM prompt

    return summary_text


    