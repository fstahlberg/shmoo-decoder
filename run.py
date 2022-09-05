from absl import app
from absl import flags

from shmoo.core import api

FLAGS = flags.FLAGS
flags.DEFINE_string("foo", "bar", "Your name.")
flags.DEFINE_integer("bla", 1,
                     "Number of times to print greeting.")

def main(argv):
  del argv  # Unused.
  
  decoder = api.Decoder()
  shmoo = api.Shmoo(decoder)
  shmoo.decode()

if __name__ == "__main__":
  app.run(main)
 
	
