import numpy as np
import sys


# You can also import his python and call the following function
# to generate the submission file
def writeSubmissionFile(labels, fn):
    f = open(fn, 'w')
    f.write("id,category\n")
    for ii, ll in enumerate(labels):
        f.write(str(ii) + "," + str(int(ll)) + "\n")

    f.close()



def main():

    if len(sys.argv) < 3:
        print("Needs 2 argument: 1) numpy file containing your predictions (same order as test.npy)\
            and 2) the output submission file name")
        sys.exit(1)

    labels = np.load(sys.argv[1])
    fn = sys.argv[2]
    writeSubmissionFile(labels, fn)


if __name__ == "__main__":
    main()    



