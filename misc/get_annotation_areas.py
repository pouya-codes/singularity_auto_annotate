import os
import os.path
from submodule_utils.metadata.annotation import *

annotations_path = 'auto_annotate/tests/mock/annotations/'
vec = []
for filename in os.listdir(annotations_path):
    annotation_path = os.path.join(annotations_path, filename)
    ga = GroovyAnnotation(annotation_path)
    area = ga.get_area(factor=1/1024**2)
    total = (area['Tumor'] if 'Tumor' in area else 0) \
            + (area['Stroma'] if 'Stroma' in area else 0)
    vec.append((ga.slide_name, area, total,))
vec = list(filter(lambda a: 'Tumor' in a[1] and 'Stroma' in a[1], vec))
vec.sort(key=lambda a: a[2])

for a in vec:
    print(*a)
