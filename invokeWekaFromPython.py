# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# *************************************************************************

# invokeWekaFromPython - adaptation from the WekaMOOC lesson
# presented in the video <https://www.youtube.com/watch?v=YT72KkkfD3w> (accessed in 11/23/2021)


import os
import traceback
import weka.core.jvm as jvm
import weka.core.converters as conv
from weka.classifiers import Evaluation, Classifier
from weka.core.classes import Random

def nomeFlor(IndicePrevisto):
    if (IndicePrevisto==0.0):
        return "0:Iris-setosa"
    if (IndicePrevisto==1.0):
        return "1:Iris-versicolor"
    else:
        return "2:Iris-virginica"

def getParameters():
    abortTrue=False
    sepalLength=0.0
    sepalWidth=0.0
    petalLength=0.0
    petalWidth=0.0
    print()
    print()
    sepalLength=float(input("Enter the sepal length in cm (range from 3.0 cm to 10.0 cm): ") or 3.0)
    if ( sepalLength < 3.0 or sepalLength > 10.0):
        abortTrue=True
    if (abortTrue==False):
        sepalWidth=float(input("Enter the sepal width in cm (range from 1.0 cm to 5.0 cm): ") or 1.0)
        if ( sepalWidth < 1.0 or sepalWidth > 5.0):
            abortTrue=True
    if (abortTrue==False):
        petalLength=float(input("Enter the petal length in cm (range from 0.2 cm to 8.0 cm): ") or 0.5)
        if ( petalLength < 0.2 or petalLength > 8.0):
            abortTrue=True
    if (abortTrue==False):
        petalWidth=float(input("Enter the petal width in cm (range from 0.2 cm to 3.0 cm): ") or 2.0)
        if ( petalWidth < 0.2 or petalWidth > 3.0):
            abortTrue=True
                                
    return abortTrue, sepalLength, sepalWidth, petalLength, petalWidth
    
def main():
    """
    Módulo principal
    """
    boolSaida=False
    sepalLength=0.0
    sepalWidth=0.0
    petalLength=0.0
    petalWidth=0.0
    
    while (boolSaida==False):

        boolSaida, sepalLength, sepalWidth, petalLength, petalWidth=getParameters()
        
        # load a dataset
        data=conv.load_any_file("D:\Arq Progrms RCS\weka-3-6-14\data\iris.arff")
        data.class_is_last()
        nomeClasse="weka.classifiers.trees.J48"
        cls=Classifier(classname=nomeClasse, options=["-C","0.25","-M","2"])
        #cls=Classifier(classname="weka.classifiers.trees.RandomForest", options=["-I","100","-K","0","-S","1"])
        evl= Evaluation(data)
        evl.crossvalidate_model(cls,data,10,Random(1))
        print(evl.summary("=== " +nomeClasse+ " on iris.arff (stats) ===", False))
        print(evl.matrix("=== " +nomeClasse+ " on iris.arff Confusion Matrix ==="))

        #***********
      
        cls.build_classifier(data)
                     
        from weka.core.dataset import Attribute, Instance, Instances
        # Parametros: [sepallength, sepalwidth, petallengh, petalwidth, classe (0==Iris-Setosa, 1==Iris-Versicolor, 2==Iris-Virginica)]

        # create attributes
        sepallength_att = Attribute.create_numeric("sepallength")
        sepalwidth_att = Attribute.create_numeric("sepalwidth")
        petallengh_att = Attribute.create_numeric("petallengh")
        petalwidth_att = Attribute.create_numeric("petalwidth")
        class_att = Attribute.create_nominal("class", ["Iris-setosa","Iris-versicolor", "Iris-virginica"])

        # create dataset
        #dataset = Instances.create_instances("InstanciaTeste", [sepallength_att, sepalwidth_att,petallengh_att,petalwidth_att, class_att],1)
        
        values = [sepalLength, sepalWidth, petalLength, petalWidth, 1.0]  #O último parâmetro não é levado em consideração na predição. Então, pode deixar como 0 (zero), 1 ou 2. Tanto faz. O que importa é que esteja dentro da faixa válida para as três classes de flor Íris (0,1,2)
        #values = [3.0, 4.0, 5.0, 2.0, 1]
        inst = Instance.create_instance(values)
        inst.dataset=data
        #dataset.add_instance(inst)
        #dataset.class_is_last()

        #print(dataset)
        #print(data)

        #for index, inst in enumerate(dataset):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        print("sepalLength=",sepalLength, "sepalWidth=",sepalWidth, "petalLength=",petalLength, "petalWidth=",petalWidth)
        print("Predição para os valores da instância informada: " + nomeFlor(pred) + ", class distribution=" + str(dist))
       

if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        print("Program is aborting...")
        print()
        jvm.stop()
