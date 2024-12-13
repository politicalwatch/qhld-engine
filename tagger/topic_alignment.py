from typing import Optional
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from tqdm import tqdm

from tipi_data.models.initiative import Initiative, TopicAlignment
from tipi_data.repositories.initiatives import Initiatives


def calculate_topic_alignment(id: Optional[str] = None):
    if not id:
        calculate_all_topic_alignments()
    else:
        try:
            calculate_single_topic_alignment(Initiatives.get(id=id))
        except Initiative.DoesNotExist:
            print(f"Initiative with id {id} not found")


def calculate_all_topic_alignments():
    query = {
            "tagged": {
                "$elemMatch": {"topics": {"$not": {"$size": 0}}}
                }
            }
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        initiatives = Initiatives.by_query(query)
        with tqdm(total=initiatives.count()) as progress:
            futures = []
            for initiative in initiatives:
                future = pool.submit(calculate_single_topic_alignment, initiative)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            for future in futures:
                future.result()


def calculate_single_topic_alignment(initiative: Initiative, needs_to_be_saved: bool = True):
    try:
        for kb in initiative['tagged']:
            topic_alignment = []
            topic_times = Counter()
            for tag in kb['tags']:
                topic_times[tag['topic']] += tag['times']
            total_times = sum(topic_times.values())
            topic_percentages = {topic: (times / total_times) * 100 for topic, times in topic_times.items()}
            for topic, percentage in topic_percentages.items():
                topic_alignment.append(
                        TopicAlignment(
                            topic=topic,
                            percentage=f"{percentage:.2f}"
                            )
                        )
            kb['topic_alignment'] = sorted(
                    topic_alignment,
                    key=lambda element: float(element['percentage']),
                    reverse=True)
                
        if needs_to_be_saved:
            initiative.save()

    except Exception as e:
        print(e)
