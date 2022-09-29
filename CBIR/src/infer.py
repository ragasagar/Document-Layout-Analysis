# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer_subset
from DB import Database

''' from color import Color
from daisy import Daisy
from edge  import Edge
from gabor import Gabor
from HOG   import HOG
from vggnet import VGGNetFeat '''
from resnet import ResNetFeat

depth = 5
d_type = 'd1'
query_idx = int(input('Query index?: '))

if __name__ == '__main__':
  db = Database()

  ''' # retrieve by color
  method = Color()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by daisy
  method = Daisy()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by edge
  method = Edge()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by gabor
  method = Gabor()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by HOG
  method = HOG()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result)

  # retrieve by VGG
  method = VGGNetFeat()
  samples = method.make_samples(db)
  query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  print(result) '''

  # retrieve by resnet
  print('CBIR using resnet')
  method = ResNetFeat()
  samples = method.make_samples(db)
  
  #query = samples[query_idx]
  #ap, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  
  query=[]
  query.append(samples[0])
  query.append(samples[2])
  query.append(samples[1])
  query.append(samples[3])
  query.append(samples[4])
  ap, result = infer_subset(query, samples=samples, depth=depth, d_type=d_type)
  
  print('Query')
  print(query)
  print('Result')
  print(result)
  print('AP')
  print(ap)
  

