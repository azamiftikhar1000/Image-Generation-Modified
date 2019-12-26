Semantic Image Manipulation:

We will use a novel hierarchical framework for semantic image manipulation. Key to the hierarchical framework is that we employ structured semantic layout as our intermediate representation for manipulation. Initialized with coarse-level bounding boxes, our structure generator first creates pixel-wise semantic layout capturing the object shape, object-object interactions, and object-scene relations. Then our image generator fills in the pixel-level textures guided by the semantic layout. Such framework allows a
user to manipulate images at object-level by adding, removing, and moving one
bounding box at a time. It performs better
than existing image generation and context hole-filing models, both qualitatively and quantitatively. Benefits of the hierarchical framework lie in applications such as semantic object manipulation, interactive image editing, and data-driven image manipulation.

It will have two main components:

1.	Structure Generator: 
2.	Image Generator:

Currently, the model is trained on Cityscape dataset. We will use transfer learning and then train the model using CoCo dataset.


<b>For more details, refer to "Image Generator Architecture.docx" file </b>
