import compressai
print("CompressAI version:", compressai.__version__)

import compressai.zoo as zoo
print("Zoo contents:", [x for x in dir(zoo) if not x.startswith("_")])

import compressai.models as models
print("Models contents:", [x for x in dir(models) if not x.startswith("_")])
