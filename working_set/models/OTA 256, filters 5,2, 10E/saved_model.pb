ї
ї7Ь7
:
Add
x"T
y"T
z"T"
Ttype:
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(

BatchDatasetV2
input_dataset

batch_size	
drop_remainder


handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
З
FlatMapDataset
input_dataset
other_arguments2
Targuments

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype

IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0
C
IteratorToStringHandle
resource_handle
string_handle


IteratorV2

handle"
shared_namestring"
	containerstring"
output_types
list(type)(0" 
output_shapeslist(shape)(0
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
,
MakeIterator
dataset
iterator
ћ

MapDataset
input_dataset
other_arguments2
Targuments

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0"$
use_inter_op_parallelismbool(" 
preserve_cardinalitybool( 
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
o
ModelDataset
input_dataset

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp

OptimizeDataset
input_dataset
optimizations

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

PrefetchDataset
input_dataset
buffer_size	

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Н
ShuffleDataset
input_dataset
buffer_size	
seed		
seed2	

handle"$
reshuffle_each_iterationbool("
output_types
list(type)(0" 
output_shapeslist(shape)(0
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

TensorSliceDataset

components2Toutput_types

handle"
Toutput_types
list(type)(0" 
output_shapeslist(shape)(0
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"fuck*1.13.12b'v1.13.1-0-g6612da8951'йн
І
ConstConst*№
valueцBу Bм../../data_exploration/datasets/32PSK_16APSK_32QAM_FM_GMSK_32APSK_OQPSK_8ASK_BPSK_8PSK_AM-SSB-SC_4ASK_16PSK_64APSK_128QAM_128APSK_AM-DSB-SC_AM-SSB-WC_64QAM_QPSK_256QAM_AM-DSB-WC_OOK_16QAM-20_-10_0_10_20_30_train.tfrecord*
dtype0*
_output_shapes
: 
g
flat_filenames/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
i
flat_filenamesReshapeConstflat_filenames/shape*
Tshape0*
_output_shapes
:*
T0
N
buffer_sizeConst*
value
B	 R'*
dtype0	*
_output_shapes
: 
F
seedConst*
dtype0	*
_output_shapes
: *
value	B	 R 
G
seed2Const*
value	B	 R *
dtype0	*
_output_shapes
: 
L

batch_sizeConst*
value	B	 Rd*
dtype0	*
_output_shapes
: 
P
drop_remainderConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
O
buffer_size_1Const*
dtype0	*
_output_shapes
: *
value	B	 Rd
Ї
Const_1Const*я
valueхBт Bл../../data_exploration/datasets/32PSK_16APSK_32QAM_FM_GMSK_32APSK_OQPSK_8ASK_BPSK_8PSK_AM-SSB-SC_4ASK_16PSK_64APSK_128QAM_128APSK_AM-DSB-SC_AM-SSB-WC_64QAM_QPSK_256QAM_AM-DSB-WC_OOK_16QAM-20_-10_0_10_20_30_test.tfrecord*
dtype0*
_output_shapes
: 
i
flat_filenames_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
o
flat_filenames_1ReshapeConst_1flat_filenames_1/shape*
T0*
Tshape0*
_output_shapes
:
N
batch_size_1Const*
_output_shapes
: *
value	B	 Rd*
dtype0	
R
drop_remainder_1Const*
value	B
 Z *
dtype0
*
_output_shapes
: 
O
buffer_size_2Const*
value	B	 Rd*
dtype0	*
_output_shapes
: 
r
x_placeholderPlaceholder*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
p
y_placeholderPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
j
ConvNet/Reshape/shapeConst*!
valueB"џџџџ      *
dtype0*
_output_shapes
:

ConvNet/ReshapeReshapex_placeholderConvNet/Reshape/shape*
T0*
Tshape0*,
_output_shapes
:џџџџџџџџџ
Е
6ConvNet/conv1d/kernel/Initializer/random_uniform/shapeConst*!
valueB"         *(
_class
loc:@ConvNet/conv1d/kernel*
dtype0*
_output_shapes
:
Ѓ
4ConvNet/conv1d/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ЄёН*(
_class
loc:@ConvNet/conv1d/kernel*
dtype0
Ѓ
4ConvNet/conv1d/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Єё=*(
_class
loc:@ConvNet/conv1d/kernel

>ConvNet/conv1d/kernel/Initializer/random_uniform/RandomUniformRandomUniform6ConvNet/conv1d/kernel/Initializer/random_uniform/shape*
dtype0*#
_output_shapes
:*

seed *
T0*(
_class
loc:@ConvNet/conv1d/kernel*
seed2 
ђ
4ConvNet/conv1d/kernel/Initializer/random_uniform/subSub4ConvNet/conv1d/kernel/Initializer/random_uniform/max4ConvNet/conv1d/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*(
_class
loc:@ConvNet/conv1d/kernel

4ConvNet/conv1d/kernel/Initializer/random_uniform/mulMul>ConvNet/conv1d/kernel/Initializer/random_uniform/RandomUniform4ConvNet/conv1d/kernel/Initializer/random_uniform/sub*
T0*(
_class
loc:@ConvNet/conv1d/kernel*#
_output_shapes
:
ћ
0ConvNet/conv1d/kernel/Initializer/random_uniformAdd4ConvNet/conv1d/kernel/Initializer/random_uniform/mul4ConvNet/conv1d/kernel/Initializer/random_uniform/min*#
_output_shapes
:*
T0*(
_class
loc:@ConvNet/conv1d/kernel
Н
ConvNet/conv1d/kernel
VariableV2*(
_class
loc:@ConvNet/conv1d/kernel*
	container *
shape:*
dtype0*#
_output_shapes
:*
shared_name 
№
ConvNet/conv1d/kernel/AssignAssignConvNet/conv1d/kernel0ConvNet/conv1d/kernel/Initializer/random_uniform*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d/kernel*
validate_shape(*#
_output_shapes
:

ConvNet/conv1d/kernel/readIdentityConvNet/conv1d/kernel*#
_output_shapes
:*
T0*(
_class
loc:@ConvNet/conv1d/kernel

%ConvNet/conv1d/bias/Initializer/zerosConst*
valueB*    *&
_class
loc:@ConvNet/conv1d/bias*
dtype0*
_output_shapes	
:
Љ
ConvNet/conv1d/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *&
_class
loc:@ConvNet/conv1d/bias*
	container 
з
ConvNet/conv1d/bias/AssignAssignConvNet/conv1d/bias%ConvNet/conv1d/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*&
_class
loc:@ConvNet/conv1d/bias

ConvNet/conv1d/bias/readIdentityConvNet/conv1d/bias*
_output_shapes	
:*
T0*&
_class
loc:@ConvNet/conv1d/bias
f
ConvNet/conv1d/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
f
$ConvNet/conv1d/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
Ќ
 ConvNet/conv1d/conv1d/ExpandDims
ExpandDimsConvNet/Reshape$ConvNet/conv1d/conv1d/ExpandDims/dim*
T0*0
_output_shapes
:џџџџџџџџџ*

Tdim0
h
&ConvNet/conv1d/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
"ConvNet/conv1d/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d/kernel/read&ConvNet/conv1d/conv1d/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:

ConvNet/conv1d/conv1d/Conv2DConv2D ConvNet/conv1d/conv1d/ExpandDims"ConvNet/conv1d/conv1d/ExpandDims_1*1
_output_shapes
:џџџџџџџџџќ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

ConvNet/conv1d/conv1d/SqueezeSqueezeConvNet/conv1d/conv1d/Conv2D*-
_output_shapes
:џџџџџџџџџќ*
squeeze_dims
*
T0
Љ
ConvNet/conv1d/BiasAddBiasAddConvNet/conv1d/conv1d/SqueezeConvNet/conv1d/bias/read*
T0*
data_formatNHWC*-
_output_shapes
:џџџџџџџџџќ
k
ConvNet/conv1d/ReluReluConvNet/conv1d/BiasAdd*
T0*-
_output_shapes
:џџџџџџџџџќ
f
$ConvNet/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
Б
 ConvNet/max_pooling1d/ExpandDims
ExpandDimsConvNet/conv1d/Relu$ConvNet/max_pooling1d/ExpandDims/dim*

Tdim0*
T0*1
_output_shapes
:џџџџџџџџџќ
й
ConvNet/max_pooling1d/MaxPoolMaxPool ConvNet/max_pooling1d/ExpandDims*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџў

ConvNet/max_pooling1d/SqueezeSqueezeConvNet/max_pooling1d/MaxPool*
squeeze_dims
*
T0*-
_output_shapes
:џџџџџџџџџў
Й
8ConvNet/conv1d_1/kernel/Initializer/random_uniform/shapeConst*!
valueB"         **
_class 
loc:@ConvNet/conv1d_1/kernel*
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *јKFН**
_class 
loc:@ConvNet/conv1d_1/kernel*
dtype0*
_output_shapes
: 
Ї
6ConvNet/conv1d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *јKF=**
_class 
loc:@ConvNet/conv1d_1/kernel*
dtype0*
_output_shapes
: 

@ConvNet/conv1d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform8ConvNet/conv1d_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*$
_output_shapes
:*

seed *
T0**
_class 
loc:@ConvNet/conv1d_1/kernel
њ
6ConvNet/conv1d_1/kernel/Initializer/random_uniform/subSub6ConvNet/conv1d_1/kernel/Initializer/random_uniform/max6ConvNet/conv1d_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*
_output_shapes
: 

6ConvNet/conv1d_1/kernel/Initializer/random_uniform/mulMul@ConvNet/conv1d_1/kernel/Initializer/random_uniform/RandomUniform6ConvNet/conv1d_1/kernel/Initializer/random_uniform/sub*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel

2ConvNet/conv1d_1/kernel/Initializer/random_uniformAdd6ConvNet/conv1d_1/kernel/Initializer/random_uniform/mul6ConvNet/conv1d_1/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*$
_output_shapes
:
У
ConvNet/conv1d_1/kernel
VariableV2*
shared_name **
_class 
loc:@ConvNet/conv1d_1/kernel*
	container *
shape:*
dtype0*$
_output_shapes
:
љ
ConvNet/conv1d_1/kernel/AssignAssignConvNet/conv1d_1/kernel2ConvNet/conv1d_1/kernel/Initializer/random_uniform*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel

ConvNet/conv1d_1/kernel/readIdentityConvNet/conv1d_1/kernel*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*$
_output_shapes
:
 
'ConvNet/conv1d_1/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *(
_class
loc:@ConvNet/conv1d_1/bias*
dtype0
­
ConvNet/conv1d_1/bias
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
п
ConvNet/conv1d_1/bias/AssignAssignConvNet/conv1d_1/bias'ConvNet/conv1d_1/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_1/bias*
validate_shape(*
_output_shapes	
:

ConvNet/conv1d_1/bias/readIdentityConvNet/conv1d_1/bias*
T0*(
_class
loc:@ConvNet/conv1d_1/bias*
_output_shapes	
:
h
ConvNet/conv1d_1/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
h
&ConvNet/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
П
"ConvNet/conv1d_1/conv1d/ExpandDims
ExpandDimsConvNet/max_pooling1d/Squeeze&ConvNet/conv1d_1/conv1d/ExpandDims/dim*
T0*1
_output_shapes
:џџџџџџџџџў*

Tdim0
j
(ConvNet/conv1d_1/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Й
$ConvNet/conv1d_1/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d_1/kernel/read(ConvNet/conv1d_1/conv1d/ExpandDims_1/dim*

Tdim0*
T0*(
_output_shapes
:

ConvNet/conv1d_1/conv1d/Conv2DConv2D"ConvNet/conv1d_1/conv1d/ExpandDims$ConvNet/conv1d_1/conv1d/ExpandDims_1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:џџџџџџџџџњ*
	dilations
*
T0

ConvNet/conv1d_1/conv1d/SqueezeSqueezeConvNet/conv1d_1/conv1d/Conv2D*
squeeze_dims
*
T0*-
_output_shapes
:џџџџџџџџџњ
Џ
ConvNet/conv1d_1/BiasAddBiasAddConvNet/conv1d_1/conv1d/SqueezeConvNet/conv1d_1/bias/read*
T0*
data_formatNHWC*-
_output_shapes
:џџџџџџџџџњ
o
ConvNet/conv1d_1/ReluReluConvNet/conv1d_1/BiasAdd*
T0*-
_output_shapes
:џџџџџџџџџњ
h
&ConvNet/max_pooling1d_1/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
З
"ConvNet/max_pooling1d_1/ExpandDims
ExpandDimsConvNet/conv1d_1/Relu&ConvNet/max_pooling1d_1/ExpandDims/dim*
T0*1
_output_shapes
:џџџџџџџџџњ*

Tdim0
н
ConvNet/max_pooling1d_1/MaxPoolMaxPool"ConvNet/max_pooling1d_1/ExpandDims*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџ§*
T0*
strides
*
data_formatNHWC

ConvNet/max_pooling1d_1/SqueezeSqueezeConvNet/max_pooling1d_1/MaxPool*-
_output_shapes
:џџџџџџџџџ§*
squeeze_dims
*
T0
Й
8ConvNet/conv1d_2/kernel/Initializer/random_uniform/shapeConst*!
valueB"         **
_class 
loc:@ConvNet/conv1d_2/kernel*
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *јKFН**
_class 
loc:@ConvNet/conv1d_2/kernel*
dtype0*
_output_shapes
: 
Ї
6ConvNet/conv1d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *јKF=**
_class 
loc:@ConvNet/conv1d_2/kernel*
dtype0*
_output_shapes
: 

@ConvNet/conv1d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform8ConvNet/conv1d_2/kernel/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
seed2 *
dtype0*$
_output_shapes
:
њ
6ConvNet/conv1d_2/kernel/Initializer/random_uniform/subSub6ConvNet/conv1d_2/kernel/Initializer/random_uniform/max6ConvNet/conv1d_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0**
_class 
loc:@ConvNet/conv1d_2/kernel

6ConvNet/conv1d_2/kernel/Initializer/random_uniform/mulMul@ConvNet/conv1d_2/kernel/Initializer/random_uniform/RandomUniform6ConvNet/conv1d_2/kernel/Initializer/random_uniform/sub*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel

2ConvNet/conv1d_2/kernel/Initializer/random_uniformAdd6ConvNet/conv1d_2/kernel/Initializer/random_uniform/mul6ConvNet/conv1d_2/kernel/Initializer/random_uniform/min*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel
У
ConvNet/conv1d_2/kernel
VariableV2*
shape:*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_2/kernel*
	container 
љ
ConvNet/conv1d_2/kernel/AssignAssignConvNet/conv1d_2/kernel2ConvNet/conv1d_2/kernel/Initializer/random_uniform*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(

ConvNet/conv1d_2/kernel/readIdentityConvNet/conv1d_2/kernel*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*$
_output_shapes
:
 
'ConvNet/conv1d_2/bias/Initializer/zerosConst*
valueB*    *(
_class
loc:@ConvNet/conv1d_2/bias*
dtype0*
_output_shapes	
:
­
ConvNet/conv1d_2/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_2/bias*
	container 
п
ConvNet/conv1d_2/bias/AssignAssignConvNet/conv1d_2/bias'ConvNet/conv1d_2/bias/Initializer/zeros*
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

ConvNet/conv1d_2/bias/readIdentityConvNet/conv1d_2/bias*
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
_output_shapes	
:
h
ConvNet/conv1d_2/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
h
&ConvNet/conv1d_2/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
С
"ConvNet/conv1d_2/conv1d/ExpandDims
ExpandDimsConvNet/max_pooling1d_1/Squeeze&ConvNet/conv1d_2/conv1d/ExpandDims/dim*

Tdim0*
T0*1
_output_shapes
:џџџџџџџџџ§
j
(ConvNet/conv1d_2/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Й
$ConvNet/conv1d_2/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d_2/kernel/read(ConvNet/conv1d_2/conv1d/ExpandDims_1/dim*(
_output_shapes
:*

Tdim0*
T0

ConvNet/conv1d_2/conv1d/Conv2DConv2D"ConvNet/conv1d_2/conv1d/ExpandDims$ConvNet/conv1d_2/conv1d/ExpandDims_1*1
_output_shapes
:џџџџџџџџџљ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

ConvNet/conv1d_2/conv1d/SqueezeSqueezeConvNet/conv1d_2/conv1d/Conv2D*
squeeze_dims
*
T0*-
_output_shapes
:џџџџџџџџџљ
Џ
ConvNet/conv1d_2/BiasAddBiasAddConvNet/conv1d_2/conv1d/SqueezeConvNet/conv1d_2/bias/read*-
_output_shapes
:џџџџџџџџџљ*
T0*
data_formatNHWC
o
ConvNet/conv1d_2/ReluReluConvNet/conv1d_2/BiasAdd*
T0*-
_output_shapes
:џџџџџџџџџљ
h
&ConvNet/max_pooling1d_2/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
З
"ConvNet/max_pooling1d_2/ExpandDims
ExpandDimsConvNet/conv1d_2/Relu&ConvNet/max_pooling1d_2/ExpandDims/dim*1
_output_shapes
:џџџџџџџџџљ*

Tdim0*
T0
н
ConvNet/max_pooling1d_2/MaxPoolMaxPool"ConvNet/max_pooling1d_2/ExpandDims*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџќ

ConvNet/max_pooling1d_2/SqueezeSqueezeConvNet/max_pooling1d_2/MaxPool*-
_output_shapes
:џџџџџџџџџќ*
squeeze_dims
*
T0
Й
8ConvNet/conv1d_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*!
valueB"         **
_class 
loc:@ConvNet/conv1d_3/kernel*
dtype0
Ї
6ConvNet/conv1d_3/kernel/Initializer/random_uniform/minConst*
valueB
 *јKFН**
_class 
loc:@ConvNet/conv1d_3/kernel*
dtype0*
_output_shapes
: 
Ї
6ConvNet/conv1d_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *јKF=**
_class 
loc:@ConvNet/conv1d_3/kernel*
dtype0*
_output_shapes
: 

@ConvNet/conv1d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform8ConvNet/conv1d_3/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*$
_output_shapes
:*

seed *
T0**
_class 
loc:@ConvNet/conv1d_3/kernel
њ
6ConvNet/conv1d_3/kernel/Initializer/random_uniform/subSub6ConvNet/conv1d_3/kernel/Initializer/random_uniform/max6ConvNet/conv1d_3/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
_output_shapes
: 

6ConvNet/conv1d_3/kernel/Initializer/random_uniform/mulMul@ConvNet/conv1d_3/kernel/Initializer/random_uniform/RandomUniform6ConvNet/conv1d_3/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*$
_output_shapes
:

2ConvNet/conv1d_3/kernel/Initializer/random_uniformAdd6ConvNet/conv1d_3/kernel/Initializer/random_uniform/mul6ConvNet/conv1d_3/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*$
_output_shapes
:
У
ConvNet/conv1d_3/kernel
VariableV2*
	container *
shape:*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_3/kernel
љ
ConvNet/conv1d_3/kernel/AssignAssignConvNet/conv1d_3/kernel2ConvNet/conv1d_3/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
validate_shape(*$
_output_shapes
:

ConvNet/conv1d_3/kernel/readIdentityConvNet/conv1d_3/kernel**
_class 
loc:@ConvNet/conv1d_3/kernel*$
_output_shapes
:*
T0
 
'ConvNet/conv1d_3/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *(
_class
loc:@ConvNet/conv1d_3/bias*
dtype0
­
ConvNet/conv1d_3/bias
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
п
ConvNet/conv1d_3/bias/AssignAssignConvNet/conv1d_3/bias'ConvNet/conv1d_3/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_3/bias

ConvNet/conv1d_3/bias/readIdentityConvNet/conv1d_3/bias*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
_output_shapes	
:
h
ConvNet/conv1d_3/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
h
&ConvNet/conv1d_3/conv1d/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
С
"ConvNet/conv1d_3/conv1d/ExpandDims
ExpandDimsConvNet/max_pooling1d_2/Squeeze&ConvNet/conv1d_3/conv1d/ExpandDims/dim*

Tdim0*
T0*1
_output_shapes
:џџџџџџџџџќ
j
(ConvNet/conv1d_3/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Й
$ConvNet/conv1d_3/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d_3/kernel/read(ConvNet/conv1d_3/conv1d/ExpandDims_1/dim*
T0*(
_output_shapes
:*

Tdim0

ConvNet/conv1d_3/conv1d/Conv2DConv2D"ConvNet/conv1d_3/conv1d/ExpandDims$ConvNet/conv1d_3/conv1d/ExpandDims_1*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:џџџџџџџџџј*
	dilations
*
T0*
strides
*
data_formatNHWC

ConvNet/conv1d_3/conv1d/SqueezeSqueezeConvNet/conv1d_3/conv1d/Conv2D*
T0*-
_output_shapes
:џџџџџџџџџј*
squeeze_dims

Џ
ConvNet/conv1d_3/BiasAddBiasAddConvNet/conv1d_3/conv1d/SqueezeConvNet/conv1d_3/bias/read*
T0*
data_formatNHWC*-
_output_shapes
:џџџџџџџџџј
o
ConvNet/conv1d_3/ReluReluConvNet/conv1d_3/BiasAdd*
T0*-
_output_shapes
:џџџџџџџџџј
h
&ConvNet/max_pooling1d_3/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
З
"ConvNet/max_pooling1d_3/ExpandDims
ExpandDimsConvNet/conv1d_3/Relu&ConvNet/max_pooling1d_3/ExpandDims/dim*

Tdim0*
T0*1
_output_shapes
:џџџџџџџџџј
м
ConvNet/max_pooling1d_3/MaxPoolMaxPool"ConvNet/max_pooling1d_3/ExpandDims*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ|

ConvNet/max_pooling1d_3/SqueezeSqueezeConvNet/max_pooling1d_3/MaxPool*
squeeze_dims
*
T0*,
_output_shapes
:џџџџџџџџџ|
Й
8ConvNet/conv1d_4/kernel/Initializer/random_uniform/shapeConst*!
valueB"         **
_class 
loc:@ConvNet/conv1d_4/kernel*
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_4/kernel/Initializer/random_uniform/minConst*
valueB
 *qФН**
_class 
loc:@ConvNet/conv1d_4/kernel*
dtype0*
_output_shapes
: 
Ї
6ConvNet/conv1d_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *qФ=**
_class 
loc:@ConvNet/conv1d_4/kernel*
dtype0*
_output_shapes
: 

@ConvNet/conv1d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform8ConvNet/conv1d_4/kernel/Initializer/random_uniform/shape*
dtype0*$
_output_shapes
:*

seed *
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*
seed2 
њ
6ConvNet/conv1d_4/kernel/Initializer/random_uniform/subSub6ConvNet/conv1d_4/kernel/Initializer/random_uniform/max6ConvNet/conv1d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0**
_class 
loc:@ConvNet/conv1d_4/kernel

6ConvNet/conv1d_4/kernel/Initializer/random_uniform/mulMul@ConvNet/conv1d_4/kernel/Initializer/random_uniform/RandomUniform6ConvNet/conv1d_4/kernel/Initializer/random_uniform/sub*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel

2ConvNet/conv1d_4/kernel/Initializer/random_uniformAdd6ConvNet/conv1d_4/kernel/Initializer/random_uniform/mul6ConvNet/conv1d_4/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*$
_output_shapes
:
У
ConvNet/conv1d_4/kernel
VariableV2*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_4/kernel*
	container *
shape:
љ
ConvNet/conv1d_4/kernel/AssignAssignConvNet/conv1d_4/kernel2ConvNet/conv1d_4/kernel/Initializer/random_uniform*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*
validate_shape(

ConvNet/conv1d_4/kernel/readIdentityConvNet/conv1d_4/kernel*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*$
_output_shapes
:
 
'ConvNet/conv1d_4/bias/Initializer/zerosConst*
valueB*    *(
_class
loc:@ConvNet/conv1d_4/bias*
dtype0*
_output_shapes	
:
­
ConvNet/conv1d_4/bias
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_4/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
п
ConvNet/conv1d_4/bias/AssignAssignConvNet/conv1d_4/bias'ConvNet/conv1d_4/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
validate_shape(*
_output_shapes	
:

ConvNet/conv1d_4/bias/readIdentityConvNet/conv1d_4/bias*
_output_shapes	
:*
T0*(
_class
loc:@ConvNet/conv1d_4/bias
h
ConvNet/conv1d_4/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
h
&ConvNet/conv1d_4/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Р
"ConvNet/conv1d_4/conv1d/ExpandDims
ExpandDimsConvNet/max_pooling1d_3/Squeeze&ConvNet/conv1d_4/conv1d/ExpandDims/dim*0
_output_shapes
:џџџџџџџџџ|*

Tdim0*
T0
j
(ConvNet/conv1d_4/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Й
$ConvNet/conv1d_4/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d_4/kernel/read(ConvNet/conv1d_4/conv1d/ExpandDims_1/dim*
T0*(
_output_shapes
:*

Tdim0

ConvNet/conv1d_4/conv1d/Conv2DConv2D"ConvNet/conv1d_4/conv1d/ExpandDims$ConvNet/conv1d_4/conv1d/ExpandDims_1*
paddingVALID*0
_output_shapes
:џџџџџџџџџ{*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

ConvNet/conv1d_4/conv1d/SqueezeSqueezeConvNet/conv1d_4/conv1d/Conv2D*,
_output_shapes
:џџџџџџџџџ{*
squeeze_dims
*
T0
Ў
ConvNet/conv1d_4/BiasAddBiasAddConvNet/conv1d_4/conv1d/SqueezeConvNet/conv1d_4/bias/read*
T0*
data_formatNHWC*,
_output_shapes
:џџџџџџџџџ{
n
ConvNet/conv1d_4/ReluReluConvNet/conv1d_4/BiasAdd*
T0*,
_output_shapes
:џџџџџџџџџ{
h
&ConvNet/max_pooling1d_4/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
"ConvNet/max_pooling1d_4/ExpandDims
ExpandDimsConvNet/conv1d_4/Relu&ConvNet/max_pooling1d_4/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:џџџџџџџџџ{
м
ConvNet/max_pooling1d_4/MaxPoolMaxPool"ConvNet/max_pooling1d_4/ExpandDims*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ=*
T0*
strides
*
data_formatNHWC

ConvNet/max_pooling1d_4/SqueezeSqueezeConvNet/max_pooling1d_4/MaxPool*
squeeze_dims
*
T0*,
_output_shapes
:џџџџџџџџџ=
Й
8ConvNet/conv1d_5/kernel/Initializer/random_uniform/shapeConst*!
valueB"         **
_class 
loc:@ConvNet/conv1d_5/kernel*
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_5/kernel/Initializer/random_uniform/minConst*
valueB
 *qФН**
_class 
loc:@ConvNet/conv1d_5/kernel*
dtype0*
_output_shapes
: 
Ї
6ConvNet/conv1d_5/kernel/Initializer/random_uniform/maxConst*
valueB
 *qФ=**
_class 
loc:@ConvNet/conv1d_5/kernel*
dtype0*
_output_shapes
: 

@ConvNet/conv1d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform8ConvNet/conv1d_5/kernel/Initializer/random_uniform/shape*
dtype0*$
_output_shapes
:*

seed *
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*
seed2 
њ
6ConvNet/conv1d_5/kernel/Initializer/random_uniform/subSub6ConvNet/conv1d_5/kernel/Initializer/random_uniform/max6ConvNet/conv1d_5/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0**
_class 
loc:@ConvNet/conv1d_5/kernel

6ConvNet/conv1d_5/kernel/Initializer/random_uniform/mulMul@ConvNet/conv1d_5/kernel/Initializer/random_uniform/RandomUniform6ConvNet/conv1d_5/kernel/Initializer/random_uniform/sub*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel

2ConvNet/conv1d_5/kernel/Initializer/random_uniformAdd6ConvNet/conv1d_5/kernel/Initializer/random_uniform/mul6ConvNet/conv1d_5/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*$
_output_shapes
:
У
ConvNet/conv1d_5/kernel
VariableV2*
shared_name **
_class 
loc:@ConvNet/conv1d_5/kernel*
	container *
shape:*
dtype0*$
_output_shapes
:
љ
ConvNet/conv1d_5/kernel/AssignAssignConvNet/conv1d_5/kernel2ConvNet/conv1d_5/kernel/Initializer/random_uniform*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*
validate_shape(

ConvNet/conv1d_5/kernel/readIdentityConvNet/conv1d_5/kernel*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*$
_output_shapes
:
 
'ConvNet/conv1d_5/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *(
_class
loc:@ConvNet/conv1d_5/bias
­
ConvNet/conv1d_5/bias
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_5/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
п
ConvNet/conv1d_5/bias/AssignAssignConvNet/conv1d_5/bias'ConvNet/conv1d_5/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_5/bias*
validate_shape(*
_output_shapes	
:

ConvNet/conv1d_5/bias/readIdentityConvNet/conv1d_5/bias*
T0*(
_class
loc:@ConvNet/conv1d_5/bias*
_output_shapes	
:
h
ConvNet/conv1d_5/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
h
&ConvNet/conv1d_5/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Р
"ConvNet/conv1d_5/conv1d/ExpandDims
ExpandDimsConvNet/max_pooling1d_4/Squeeze&ConvNet/conv1d_5/conv1d/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:џџџџџџџџџ=
j
(ConvNet/conv1d_5/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Й
$ConvNet/conv1d_5/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d_5/kernel/read(ConvNet/conv1d_5/conv1d/ExpandDims_1/dim*

Tdim0*
T0*(
_output_shapes
:

ConvNet/conv1d_5/conv1d/Conv2DConv2D"ConvNet/conv1d_5/conv1d/ExpandDims$ConvNet/conv1d_5/conv1d/ExpandDims_1*0
_output_shapes
:џџџџџџџџџ<*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

ConvNet/conv1d_5/conv1d/SqueezeSqueezeConvNet/conv1d_5/conv1d/Conv2D*
T0*,
_output_shapes
:џџџџџџџџџ<*
squeeze_dims

Ў
ConvNet/conv1d_5/BiasAddBiasAddConvNet/conv1d_5/conv1d/SqueezeConvNet/conv1d_5/bias/read*
T0*
data_formatNHWC*,
_output_shapes
:џџџџџџџџџ<
n
ConvNet/conv1d_5/ReluReluConvNet/conv1d_5/BiasAdd*,
_output_shapes
:џџџџџџџџџ<*
T0
h
&ConvNet/max_pooling1d_5/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
Ж
"ConvNet/max_pooling1d_5/ExpandDims
ExpandDimsConvNet/conv1d_5/Relu&ConvNet/max_pooling1d_5/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:џџџџџџџџџ<
м
ConvNet/max_pooling1d_5/MaxPoolMaxPool"ConvNet/max_pooling1d_5/ExpandDims*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ

ConvNet/max_pooling1d_5/SqueezeSqueezeConvNet/max_pooling1d_5/MaxPool*
T0*,
_output_shapes
:џџџџџџџџџ*
squeeze_dims

Й
8ConvNet/conv1d_6/kernel/Initializer/random_uniform/shapeConst*!
valueB"         **
_class 
loc:@ConvNet/conv1d_6/kernel*
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_6/kernel/Initializer/random_uniform/minConst*
valueB
 *qФН**
_class 
loc:@ConvNet/conv1d_6/kernel*
dtype0*
_output_shapes
: 
Ї
6ConvNet/conv1d_6/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *qФ=**
_class 
loc:@ConvNet/conv1d_6/kernel

@ConvNet/conv1d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform8ConvNet/conv1d_6/kernel/Initializer/random_uniform/shape*$
_output_shapes
:*

seed *
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
seed2 *
dtype0
њ
6ConvNet/conv1d_6/kernel/Initializer/random_uniform/subSub6ConvNet/conv1d_6/kernel/Initializer/random_uniform/max6ConvNet/conv1d_6/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
_output_shapes
: 

6ConvNet/conv1d_6/kernel/Initializer/random_uniform/mulMul@ConvNet/conv1d_6/kernel/Initializer/random_uniform/RandomUniform6ConvNet/conv1d_6/kernel/Initializer/random_uniform/sub*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*$
_output_shapes
:

2ConvNet/conv1d_6/kernel/Initializer/random_uniformAdd6ConvNet/conv1d_6/kernel/Initializer/random_uniform/mul6ConvNet/conv1d_6/kernel/Initializer/random_uniform/min*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*$
_output_shapes
:
У
ConvNet/conv1d_6/kernel
VariableV2*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_6/kernel*
	container *
shape:
љ
ConvNet/conv1d_6/kernel/AssignAssignConvNet/conv1d_6/kernel2ConvNet/conv1d_6/kernel/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
validate_shape(*$
_output_shapes
:

ConvNet/conv1d_6/kernel/readIdentityConvNet/conv1d_6/kernel*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*$
_output_shapes
:
 
'ConvNet/conv1d_6/bias/Initializer/zerosConst*
valueB*    *(
_class
loc:@ConvNet/conv1d_6/bias*
dtype0*
_output_shapes	
:
­
ConvNet/conv1d_6/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_6/bias*
	container 
п
ConvNet/conv1d_6/bias/AssignAssignConvNet/conv1d_6/bias'ConvNet/conv1d_6/bias/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
validate_shape(*
_output_shapes	
:

ConvNet/conv1d_6/bias/readIdentityConvNet/conv1d_6/bias*
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
_output_shapes	
:
h
ConvNet/conv1d_6/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
h
&ConvNet/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
value	B :*
dtype0
Р
"ConvNet/conv1d_6/conv1d/ExpandDims
ExpandDimsConvNet/max_pooling1d_5/Squeeze&ConvNet/conv1d_6/conv1d/ExpandDims/dim*

Tdim0*
T0*0
_output_shapes
:џџџџџџџџџ
j
(ConvNet/conv1d_6/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Й
$ConvNet/conv1d_6/conv1d/ExpandDims_1
ExpandDimsConvNet/conv1d_6/kernel/read(ConvNet/conv1d_6/conv1d/ExpandDims_1/dim*
T0*(
_output_shapes
:*

Tdim0

ConvNet/conv1d_6/conv1d/Conv2DConv2D"ConvNet/conv1d_6/conv1d/ExpandDims$ConvNet/conv1d_6/conv1d/ExpandDims_1*
paddingVALID*0
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

ConvNet/conv1d_6/conv1d/SqueezeSqueezeConvNet/conv1d_6/conv1d/Conv2D*,
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0
Ў
ConvNet/conv1d_6/BiasAddBiasAddConvNet/conv1d_6/conv1d/SqueezeConvNet/conv1d_6/bias/read*
T0*
data_formatNHWC*,
_output_shapes
:џџџџџџџџџ
n
ConvNet/conv1d_6/ReluReluConvNet/conv1d_6/BiasAdd*,
_output_shapes
:џџџџџџџџџ*
T0
h
&ConvNet/max_pooling1d_6/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
Ж
"ConvNet/max_pooling1d_6/ExpandDims
ExpandDimsConvNet/conv1d_6/Relu&ConvNet/max_pooling1d_6/ExpandDims/dim*0
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
м
ConvNet/max_pooling1d_6/MaxPoolMaxPool"ConvNet/max_pooling1d_6/ExpandDims*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ

ConvNet/max_pooling1d_6/SqueezeSqueezeConvNet/max_pooling1d_6/MaxPool*,
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0
|
ConvNet/Flatten/flatten/ShapeShapeConvNet/max_pooling1d_6/Squeeze*
_output_shapes
:*
T0*
out_type0
u
+ConvNet/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-ConvNet/Flatten/flatten/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
w
-ConvNet/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ё
%ConvNet/Flatten/flatten/strided_sliceStridedSliceConvNet/Flatten/flatten/Shape+ConvNet/Flatten/flatten/strided_slice/stack-ConvNet/Flatten/flatten/strided_slice/stack_1-ConvNet/Flatten/flatten/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
r
'ConvNet/Flatten/flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
З
%ConvNet/Flatten/flatten/Reshape/shapePack%ConvNet/Flatten/flatten/strided_slice'ConvNet/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Г
ConvNet/Flatten/flatten/ReshapeReshapeConvNet/max_pooling1d_6/Squeeze%ConvNet/Flatten/flatten/Reshape/shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Џ
5ConvNet/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *'
_class
loc:@ConvNet/dense/kernel
Ё
3ConvNet/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *.­$Н*'
_class
loc:@ConvNet/dense/kernel
Ё
3ConvNet/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *.­$=*'
_class
loc:@ConvNet/dense/kernel*
dtype0*
_output_shapes
: 
џ
=ConvNet/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5ConvNet/dense/kernel/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@ConvNet/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:

ю
3ConvNet/dense/kernel/Initializer/random_uniform/subSub3ConvNet/dense/kernel/Initializer/random_uniform/max3ConvNet/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@ConvNet/dense/kernel

3ConvNet/dense/kernel/Initializer/random_uniform/mulMul=ConvNet/dense/kernel/Initializer/random_uniform/RandomUniform3ConvNet/dense/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*'
_class
loc:@ConvNet/dense/kernel
є
/ConvNet/dense/kernel/Initializer/random_uniformAdd3ConvNet/dense/kernel/Initializer/random_uniform/mul3ConvNet/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:

Е
ConvNet/dense/kernel
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

щ
ConvNet/dense/kernel/AssignAssignConvNet/dense/kernel/ConvNet/dense/kernel/Initializer/random_uniform*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

ConvNet/dense/kernel/readIdentityConvNet/dense/kernel*
T0*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:


$ConvNet/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *%
_class
loc:@ConvNet/dense/bias
Ї
ConvNet/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *%
_class
loc:@ConvNet/dense/bias*
	container *
shape:
г
ConvNet/dense/bias/AssignAssignConvNet/dense/bias$ConvNet/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense/bias/readIdentityConvNet/dense/bias*
T0*%
_class
loc:@ConvNet/dense/bias*
_output_shapes	
:
Г
ConvNet/dense/MatMulMatMulConvNet/Flatten/flatten/ReshapeConvNet/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

ConvNet/dense/BiasAddBiasAddConvNet/dense/MatMulConvNet/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
d
ConvNet/dense/ReluReluConvNet/dense/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Г
7ConvNet/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_1/kernel*
dtype0*
_output_shapes
:
Ѕ
5ConvNet/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *qФО*)
_class
loc:@ConvNet/dense_1/kernel*
dtype0*
_output_shapes
: 
Ѕ
5ConvNet/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *qФ>*)
_class
loc:@ConvNet/dense_1/kernel*
dtype0*
_output_shapes
: 

?ConvNet/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_1/kernel
і
5ConvNet/dense_1/kernel/Initializer/random_uniform/subSub5ConvNet/dense_1/kernel/Initializer/random_uniform/max5ConvNet/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
_output_shapes
: 

5ConvNet/dense_1/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_1/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:

ќ
1ConvNet/dense_1/kernel/Initializer/random_uniformAdd5ConvNet/dense_1/kernel/Initializer/random_uniform/mul5ConvNet/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:

Й
ConvNet/dense_1/kernel
VariableV2*)
_class
loc:@ConvNet/dense_1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ё
ConvNet/dense_1/kernel/AssignAssignConvNet/dense_1/kernel1ConvNet/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(* 
_output_shapes
:


ConvNet/dense_1/kernel/readIdentityConvNet/dense_1/kernel*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:


&ConvNet/dense_1/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@ConvNet/dense_1/bias*
dtype0*
_output_shapes	
:
Ћ
ConvNet/dense_1/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_1/bias
л
ConvNet/dense_1/bias/AssignAssignConvNet/dense_1/bias&ConvNet/dense_1/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(

ConvNet/dense_1/bias/readIdentityConvNet/dense_1/bias*
T0*'
_class
loc:@ConvNet/dense_1/bias*
_output_shapes	
:
Њ
ConvNet/dense_1/MatMulMatMulConvNet/dense/ReluConvNet/dense_1/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

ConvNet/dense_1/BiasAddBiasAddConvNet/dense_1/MatMulConvNet/dense_1/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
h
ConvNet/dense_1/ReluReluConvNet/dense_1/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
С
>ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
dtype0
Г
<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *ЛrKО*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
dtype0
Г
<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/maxConst*
valueB
 *ЛrK>*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
dtype0*
_output_shapes
: 

FConvNet/cnn_logits_out/kernel/Initializer/random_uniform/RandomUniformRandomUniform>ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/shape*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 

<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/subSub<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/max<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/min*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
: *
T0
Ѕ
<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/mulMulFConvNet/cnn_logits_out/kernel/Initializer/random_uniform/RandomUniform<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	

8ConvNet/cnn_logits_out/kernel/Initializer/random_uniformAdd<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/mul<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel
Х
ConvNet/cnn_logits_out/kernel
VariableV2*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 

$ConvNet/cnn_logits_out/kernel/AssignAssignConvNet/cnn_logits_out/kernel8ConvNet/cnn_logits_out/kernel/Initializer/random_uniform*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
Љ
"ConvNet/cnn_logits_out/kernel/readIdentityConvNet/cnn_logits_out/kernel*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	
Њ
-ConvNet/cnn_logits_out/bias/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
dtype0*
_output_shapes
:
З
ConvNet/cnn_logits_out/bias
VariableV2*
_output_shapes
:*
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape:*
dtype0
і
"ConvNet/cnn_logits_out/bias/AssignAssignConvNet/cnn_logits_out/bias-ConvNet/cnn_logits_out/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(

 ConvNet/cnn_logits_out/bias/readIdentityConvNet/cnn_logits_out/bias*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
:*
T0
Й
ConvNet/cnn_logits_out/MatMulMatMulConvNet/dense_1/Relu"ConvNet/cnn_logits_out/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Г
ConvNet/cnn_logits_out/BiasAddBiasAddConvNet/cnn_logits_out/MatMul ConvNet/cnn_logits_out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ

9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradienty_placeholder*
T0*'
_output_shapes
:џџџџџџџџџ
k
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 

*softmax_cross_entropy_with_logits_sg/ShapeShapeConvNet/cnn_logits_out/BiasAdd*
T0*
out_type0*
_output_shapes
:
m
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

,softmax_cross_entropy_with_logits_sg/Shape_1ShapeConvNet/cnn_logits_out/BiasAdd*
T0*
out_type0*
_output_shapes
:
l
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Љ
(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 

0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
y
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
і
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
r
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Э
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeConvNet/cnn_logits_out/BiasAdd+softmax_cross_entropy_with_logits_sg/concat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
Ѕ
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
_output_shapes
:*
T0*
out_type0
n
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
­
*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
 
2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:
{
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
ќ
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
t
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
ь
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
э
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Ћ
*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
|
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:

1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
њ
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
_output_shapes
:*
Index0*
T0
Щ
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
Tshape0*#
_output_shapes
:џџџџџџџџџ*
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:

MeanMean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/Mean_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
Ї
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
ю
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ

gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
с
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
Е
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
Й
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 

@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
і
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Т
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
п
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
х
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1

Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeConvNet/cnn_logits_out/BiasAdd*
_output_shapes
:*
T0*
out_type0

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
9gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Ш
>gradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/group_depsNoOp:^gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGradD^gradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
т
Fgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape?^gradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/group_deps*V
_classL
JHloc:@gradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
У
Hgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGrad?^gradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGrad

3gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMulMatMulFgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency"ConvNet/cnn_logits_out/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
э
5gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1MatMulConvNet/dense_1/ReluFgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Г
=gradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/group_depsNoOp4^gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul6^gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1
С
Egradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependencyIdentity3gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul>^gradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
О
Ggradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependency_1Identity5gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1>^gradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1*
_output_shapes
:	
Ш
,gradients/ConvNet/dense_1/Relu_grad/ReluGradReluGradEgradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependencyConvNet/dense_1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
Ќ
2gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ѓ
7gradients/ConvNet/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_1/Relu_grad/ReluGrad
Ї
?gradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_1/Relu_grad/ReluGrad8^gradients/ConvNet/dense_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
Ј
Agradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_1/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
э
,gradients/ConvNet/dense_1/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependencyConvNet/dense_1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
о
.gradients/ConvNet/dense_1/MatMul_grad/MatMul_1MatMulConvNet/dense/Relu?gradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

6gradients/ConvNet/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_1/MatMul_grad/MatMul/^gradients/ConvNet/dense_1/MatMul_grad/MatMul_1
Ѕ
>gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_1/MatMul_grad/MatMul7^gradients/ConvNet/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/ConvNet/dense_1/MatMul_grad/MatMul
Ѓ
@gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_1/MatMul_grad/MatMul_17^gradients/ConvNet/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/ConvNet/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

Н
*gradients/ConvNet/dense/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependencyConvNet/dense/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
Ј
0gradients/ConvNet/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/ConvNet/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0

5gradients/ConvNet/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/ConvNet/dense/BiasAdd_grad/BiasAddGrad+^gradients/ConvNet/dense/Relu_grad/ReluGrad

=gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/ConvNet/dense/Relu_grad/ReluGrad6^gradients/ConvNet/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/ConvNet/dense/Relu_grad/ReluGrad*(
_output_shapes
:џџџџџџџџџ
 
?gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/ConvNet/dense/BiasAdd_grad/BiasAddGrad6^gradients/ConvNet/dense/BiasAdd_grad/tuple/group_deps*C
_class9
75loc:@gradients/ConvNet/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ч
*gradients/ConvNet/dense/MatMul_grad/MatMulMatMul=gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependencyConvNet/dense/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ч
,gradients/ConvNet/dense/MatMul_grad/MatMul_1MatMulConvNet/Flatten/flatten/Reshape=gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

4gradients/ConvNet/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/ConvNet/dense/MatMul_grad/MatMul-^gradients/ConvNet/dense/MatMul_grad/MatMul_1

<gradients/ConvNet/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/ConvNet/dense/MatMul_grad/MatMul5^gradients/ConvNet/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/ConvNet/dense/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

>gradients/ConvNet/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/ConvNet/dense/MatMul_grad/MatMul_15^gradients/ConvNet/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense/MatMul_grad/MatMul_1* 
_output_shapes
:


4gradients/ConvNet/Flatten/flatten/Reshape_grad/ShapeShapeConvNet/max_pooling1d_6/Squeeze*
T0*
out_type0*
_output_shapes
:
њ
6gradients/ConvNet/Flatten/flatten/Reshape_grad/ReshapeReshape<gradients/ConvNet/dense/MatMul_grad/tuple/control_dependency4gradients/ConvNet/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:џџџџџџџџџ

4gradients/ConvNet/max_pooling1d_6/Squeeze_grad/ShapeShapeConvNet/max_pooling1d_6/MaxPool*
_output_shapes
:*
T0*
out_type0
ј
6gradients/ConvNet/max_pooling1d_6/Squeeze_grad/ReshapeReshape6gradients/ConvNet/Flatten/flatten/Reshape_grad/Reshape4gradients/ConvNet/max_pooling1d_6/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџ
д
:gradients/ConvNet/max_pooling1d_6/MaxPool_grad/MaxPoolGradMaxPoolGrad"ConvNet/max_pooling1d_6/ExpandDimsConvNet/max_pooling1d_6/MaxPool6gradients/ConvNet/max_pooling1d_6/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ*
T0

7gradients/ConvNet/max_pooling1d_6/ExpandDims_grad/ShapeShapeConvNet/conv1d_6/Relu*
out_type0*
_output_shapes
:*
T0
ў
9gradients/ConvNet/max_pooling1d_6/ExpandDims_grad/ReshapeReshape:gradients/ConvNet/max_pooling1d_6/MaxPool_grad/MaxPoolGrad7gradients/ConvNet/max_pooling1d_6/ExpandDims_grad/Shape*
T0*
Tshape0*,
_output_shapes
:џџџџџџџџџ
Т
-gradients/ConvNet/conv1d_6/Relu_grad/ReluGradReluGrad9gradients/ConvNet/max_pooling1d_6/ExpandDims_grad/ReshapeConvNet/conv1d_6/Relu*,
_output_shapes
:џџџџџџџџџ*
T0
Ў
3gradients/ConvNet/conv1d_6/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/ConvNet/conv1d_6/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
І
8gradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/group_depsNoOp4^gradients/ConvNet/conv1d_6/BiasAdd_grad/BiasAddGrad.^gradients/ConvNet/conv1d_6/Relu_grad/ReluGrad
Џ
@gradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/ConvNet/conv1d_6/Relu_grad/ReluGrad9^gradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients/ConvNet/conv1d_6/Relu_grad/ReluGrad*,
_output_shapes
:џџџџџџџџџ*
T0
Ќ
Bgradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/ConvNet/conv1d_6/BiasAdd_grad/BiasAddGrad9^gradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*F
_class<
:8loc:@gradients/ConvNet/conv1d_6/BiasAdd_grad/BiasAddGrad

4gradients/ConvNet/conv1d_6/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d_6/conv1d/Conv2D*
out_type0*
_output_shapes
:*
T0

6gradients/ConvNet/conv1d_6/conv1d/Squeeze_grad/ReshapeReshape@gradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/control_dependency4gradients/ConvNet/conv1d_6/conv1d/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџ
Ь
4gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/ShapeNShapeN"ConvNet/conv1d_6/conv1d/ExpandDims$ConvNet/conv1d_6/conv1d/ExpandDims_1*
T0*
out_type0*
N* 
_output_shapes
::

Agradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput4gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/ShapeN$ConvNet/conv1d_6/conv1d/ExpandDims_16gradients/ConvNet/conv1d_6/conv1d/Squeeze_grad/Reshape*
paddingVALID*0
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(

Bgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"ConvNet/conv1d_6/conv1d/ExpandDims6gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/ShapeN:16gradients/ConvNet/conv1d_6/conv1d/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC
Я
>gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/group_depsNoOpC^gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropFilterB^gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropInput
ч
Fgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/control_dependencyIdentityAgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropInput?^gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ
у
Hgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityBgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropFilter?^gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/group_deps*(
_output_shapes
:*
T0*U
_classK
IGloc:@gradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/Conv2DBackpropFilter

7gradients/ConvNet/conv1d_6/conv1d/ExpandDims_grad/ShapeShapeConvNet/max_pooling1d_5/Squeeze*
_output_shapes
:*
T0*
out_type0

9gradients/ConvNet/conv1d_6/conv1d/ExpandDims_grad/ReshapeReshapeFgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/control_dependency7gradients/ConvNet/conv1d_6/conv1d/ExpandDims_grad/Shape*
T0*
Tshape0*,
_output_shapes
:џџџџџџџџџ

9gradients/ConvNet/conv1d_6/conv1d/ExpandDims_1_grad/ShapeConst*
dtype0*
_output_shapes
:*!
valueB"         

;gradients/ConvNet/conv1d_6/conv1d/ExpandDims_1_grad/ReshapeReshapeHgradients/ConvNet/conv1d_6/conv1d/Conv2D_grad/tuple/control_dependency_19gradients/ConvNet/conv1d_6/conv1d/ExpandDims_1_grad/Shape*
Tshape0*$
_output_shapes
:*
T0

4gradients/ConvNet/max_pooling1d_5/Squeeze_grad/ShapeShapeConvNet/max_pooling1d_5/MaxPool*
_output_shapes
:*
T0*
out_type0
ћ
6gradients/ConvNet/max_pooling1d_5/Squeeze_grad/ReshapeReshape9gradients/ConvNet/conv1d_6/conv1d/ExpandDims_grad/Reshape4gradients/ConvNet/max_pooling1d_5/Squeeze_grad/Shape*0
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
д
:gradients/ConvNet/max_pooling1d_5/MaxPool_grad/MaxPoolGradMaxPoolGrad"ConvNet/max_pooling1d_5/ExpandDimsConvNet/max_pooling1d_5/MaxPool6gradients/ConvNet/max_pooling1d_5/Squeeze_grad/Reshape*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ<

7gradients/ConvNet/max_pooling1d_5/ExpandDims_grad/ShapeShapeConvNet/conv1d_5/Relu*
out_type0*
_output_shapes
:*
T0
ў
9gradients/ConvNet/max_pooling1d_5/ExpandDims_grad/ReshapeReshape:gradients/ConvNet/max_pooling1d_5/MaxPool_grad/MaxPoolGrad7gradients/ConvNet/max_pooling1d_5/ExpandDims_grad/Shape*
T0*
Tshape0*,
_output_shapes
:џџџџџџџџџ<
Т
-gradients/ConvNet/conv1d_5/Relu_grad/ReluGradReluGrad9gradients/ConvNet/max_pooling1d_5/ExpandDims_grad/ReshapeConvNet/conv1d_5/Relu*,
_output_shapes
:џџџџџџџџџ<*
T0
Ў
3gradients/ConvNet/conv1d_5/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/ConvNet/conv1d_5/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
І
8gradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/group_depsNoOp4^gradients/ConvNet/conv1d_5/BiasAdd_grad/BiasAddGrad.^gradients/ConvNet/conv1d_5/Relu_grad/ReluGrad
Џ
@gradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/ConvNet/conv1d_5/Relu_grad/ReluGrad9^gradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/ConvNet/conv1d_5/Relu_grad/ReluGrad*,
_output_shapes
:џџџџџџџџџ<
Ќ
Bgradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/ConvNet/conv1d_5/BiasAdd_grad/BiasAddGrad9^gradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/ConvNet/conv1d_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

4gradients/ConvNet/conv1d_5/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d_5/conv1d/Conv2D*
_output_shapes
:*
T0*
out_type0

6gradients/ConvNet/conv1d_5/conv1d/Squeeze_grad/ReshapeReshape@gradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/control_dependency4gradients/ConvNet/conv1d_5/conv1d/Squeeze_grad/Shape*
Tshape0*0
_output_shapes
:џџџџџџџџџ<*
T0
Ь
4gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/ShapeNShapeN"ConvNet/conv1d_5/conv1d/ExpandDims$ConvNet/conv1d_5/conv1d/ExpandDims_1*
T0*
out_type0*
N* 
_output_shapes
::

Agradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput4gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/ShapeN$ConvNet/conv1d_5/conv1d/ExpandDims_16gradients/ConvNet/conv1d_5/conv1d/Squeeze_grad/Reshape*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:џџџџџџџџџ=*
	dilations
*
T0*
strides
*
data_formatNHWC

Bgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"ConvNet/conv1d_5/conv1d/ExpandDims6gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/ShapeN:16gradients/ConvNet/conv1d_5/conv1d/Squeeze_grad/Reshape*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
Я
>gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/group_depsNoOpC^gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropFilterB^gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropInput
ч
Fgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/control_dependencyIdentityAgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropInput?^gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ=
у
Hgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityBgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropFilter?^gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:

7gradients/ConvNet/conv1d_5/conv1d/ExpandDims_grad/ShapeShapeConvNet/max_pooling1d_4/Squeeze*
T0*
out_type0*
_output_shapes
:

9gradients/ConvNet/conv1d_5/conv1d/ExpandDims_grad/ReshapeReshapeFgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/control_dependency7gradients/ConvNet/conv1d_5/conv1d/ExpandDims_grad/Shape*,
_output_shapes
:џџџџџџџџџ=*
T0*
Tshape0

9gradients/ConvNet/conv1d_5/conv1d/ExpandDims_1_grad/ShapeConst*
_output_shapes
:*!
valueB"         *
dtype0

;gradients/ConvNet/conv1d_5/conv1d/ExpandDims_1_grad/ReshapeReshapeHgradients/ConvNet/conv1d_5/conv1d/Conv2D_grad/tuple/control_dependency_19gradients/ConvNet/conv1d_5/conv1d/ExpandDims_1_grad/Shape*$
_output_shapes
:*
T0*
Tshape0

4gradients/ConvNet/max_pooling1d_4/Squeeze_grad/ShapeShapeConvNet/max_pooling1d_4/MaxPool*
T0*
out_type0*
_output_shapes
:
ћ
6gradients/ConvNet/max_pooling1d_4/Squeeze_grad/ReshapeReshape9gradients/ConvNet/conv1d_5/conv1d/ExpandDims_grad/Reshape4gradients/ConvNet/max_pooling1d_4/Squeeze_grad/Shape*0
_output_shapes
:џџџџџџџџџ=*
T0*
Tshape0
д
:gradients/ConvNet/max_pooling1d_4/MaxPool_grad/MaxPoolGradMaxPoolGrad"ConvNet/max_pooling1d_4/ExpandDimsConvNet/max_pooling1d_4/MaxPool6gradients/ConvNet/max_pooling1d_4/Squeeze_grad/Reshape*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*0
_output_shapes
:џџџџџџџџџ{

7gradients/ConvNet/max_pooling1d_4/ExpandDims_grad/ShapeShapeConvNet/conv1d_4/Relu*
T0*
out_type0*
_output_shapes
:
ў
9gradients/ConvNet/max_pooling1d_4/ExpandDims_grad/ReshapeReshape:gradients/ConvNet/max_pooling1d_4/MaxPool_grad/MaxPoolGrad7gradients/ConvNet/max_pooling1d_4/ExpandDims_grad/Shape*,
_output_shapes
:џџџџџџџџџ{*
T0*
Tshape0
Т
-gradients/ConvNet/conv1d_4/Relu_grad/ReluGradReluGrad9gradients/ConvNet/max_pooling1d_4/ExpandDims_grad/ReshapeConvNet/conv1d_4/Relu*,
_output_shapes
:џџџџџџџџџ{*
T0
Ў
3gradients/ConvNet/conv1d_4/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/ConvNet/conv1d_4/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
І
8gradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/group_depsNoOp4^gradients/ConvNet/conv1d_4/BiasAdd_grad/BiasAddGrad.^gradients/ConvNet/conv1d_4/Relu_grad/ReluGrad
Џ
@gradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/ConvNet/conv1d_4/Relu_grad/ReluGrad9^gradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/ConvNet/conv1d_4/Relu_grad/ReluGrad*,
_output_shapes
:џџџџџџџџџ{
Ќ
Bgradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/ConvNet/conv1d_4/BiasAdd_grad/BiasAddGrad9^gradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*F
_class<
:8loc:@gradients/ConvNet/conv1d_4/BiasAdd_grad/BiasAddGrad

4gradients/ConvNet/conv1d_4/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d_4/conv1d/Conv2D*
T0*
out_type0*
_output_shapes
:

6gradients/ConvNet/conv1d_4/conv1d/Squeeze_grad/ReshapeReshape@gradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/control_dependency4gradients/ConvNet/conv1d_4/conv1d/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџ{
Ь
4gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/ShapeNShapeN"ConvNet/conv1d_4/conv1d/ExpandDims$ConvNet/conv1d_4/conv1d/ExpandDims_1*
out_type0*
N* 
_output_shapes
::*
T0

Agradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput4gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/ShapeN$ConvNet/conv1d_4/conv1d/ExpandDims_16gradients/ConvNet/conv1d_4/conv1d/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:џџџџџџџџџ|*
	dilations
*
T0

Bgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"ConvNet/conv1d_4/conv1d/ExpandDims6gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/ShapeN:16gradients/ConvNet/conv1d_4/conv1d/Squeeze_grad/Reshape*
paddingVALID*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Я
>gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/group_depsNoOpC^gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropFilterB^gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropInput
ч
Fgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/control_dependencyIdentityAgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropInput?^gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropInput*0
_output_shapes
:џџџџџџџџџ|
у
Hgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityBgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropFilter?^gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:

7gradients/ConvNet/conv1d_4/conv1d/ExpandDims_grad/ShapeShapeConvNet/max_pooling1d_3/Squeeze*
T0*
out_type0*
_output_shapes
:

9gradients/ConvNet/conv1d_4/conv1d/ExpandDims_grad/ReshapeReshapeFgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/control_dependency7gradients/ConvNet/conv1d_4/conv1d/ExpandDims_grad/Shape*
T0*
Tshape0*,
_output_shapes
:џџџџџџџџџ|

9gradients/ConvNet/conv1d_4/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:

;gradients/ConvNet/conv1d_4/conv1d/ExpandDims_1_grad/ReshapeReshapeHgradients/ConvNet/conv1d_4/conv1d/Conv2D_grad/tuple/control_dependency_19gradients/ConvNet/conv1d_4/conv1d/ExpandDims_1_grad/Shape*
T0*
Tshape0*$
_output_shapes
:

4gradients/ConvNet/max_pooling1d_3/Squeeze_grad/ShapeShapeConvNet/max_pooling1d_3/MaxPool*
T0*
out_type0*
_output_shapes
:
ћ
6gradients/ConvNet/max_pooling1d_3/Squeeze_grad/ReshapeReshape9gradients/ConvNet/conv1d_4/conv1d/ExpandDims_grad/Reshape4gradients/ConvNet/max_pooling1d_3/Squeeze_grad/Shape*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџ|
е
:gradients/ConvNet/max_pooling1d_3/MaxPool_grad/MaxPoolGradMaxPoolGrad"ConvNet/max_pooling1d_3/ExpandDimsConvNet/max_pooling1d_3/MaxPool6gradients/ConvNet/max_pooling1d_3/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџј*
T0

7gradients/ConvNet/max_pooling1d_3/ExpandDims_grad/ShapeShapeConvNet/conv1d_3/Relu*
T0*
out_type0*
_output_shapes
:
џ
9gradients/ConvNet/max_pooling1d_3/ExpandDims_grad/ReshapeReshape:gradients/ConvNet/max_pooling1d_3/MaxPool_grad/MaxPoolGrad7gradients/ConvNet/max_pooling1d_3/ExpandDims_grad/Shape*
Tshape0*-
_output_shapes
:џџџџџџџџџј*
T0
У
-gradients/ConvNet/conv1d_3/Relu_grad/ReluGradReluGrad9gradients/ConvNet/max_pooling1d_3/ExpandDims_grad/ReshapeConvNet/conv1d_3/Relu*-
_output_shapes
:џџџџџџџџџј*
T0
Ў
3gradients/ConvNet/conv1d_3/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/ConvNet/conv1d_3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
І
8gradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/group_depsNoOp4^gradients/ConvNet/conv1d_3/BiasAdd_grad/BiasAddGrad.^gradients/ConvNet/conv1d_3/Relu_grad/ReluGrad
А
@gradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/ConvNet/conv1d_3/Relu_grad/ReluGrad9^gradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/ConvNet/conv1d_3/Relu_grad/ReluGrad*-
_output_shapes
:џџџџџџџџџј
Ќ
Bgradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/ConvNet/conv1d_3/BiasAdd_grad/BiasAddGrad9^gradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/ConvNet/conv1d_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

4gradients/ConvNet/conv1d_3/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d_3/conv1d/Conv2D*
T0*
out_type0*
_output_shapes
:

6gradients/ConvNet/conv1d_3/conv1d/Squeeze_grad/ReshapeReshape@gradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/control_dependency4gradients/ConvNet/conv1d_3/conv1d/Squeeze_grad/Shape*
T0*
Tshape0*1
_output_shapes
:џџџџџџџџџј
Ь
4gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/ShapeNShapeN"ConvNet/conv1d_3/conv1d/ExpandDims$ConvNet/conv1d_3/conv1d/ExpandDims_1*
T0*
out_type0*
N* 
_output_shapes
::

Agradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput4gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/ShapeN$ConvNet/conv1d_3/conv1d/ExpandDims_16gradients/ConvNet/conv1d_3/conv1d/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:џџџџџџџџџќ*
	dilations
*
T0

Bgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"ConvNet/conv1d_3/conv1d/ExpandDims6gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/ShapeN:16gradients/ConvNet/conv1d_3/conv1d/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:*
	dilations
*
T0
Я
>gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/group_depsNoOpC^gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropFilterB^gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropInput
ш
Fgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/control_dependencyIdentityAgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropInput?^gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџќ
у
Hgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityBgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropFilter?^gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/group_deps*U
_classK
IGloc:@gradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:*
T0

7gradients/ConvNet/conv1d_3/conv1d/ExpandDims_grad/ShapeShapeConvNet/max_pooling1d_2/Squeeze*
T0*
out_type0*
_output_shapes
:

9gradients/ConvNet/conv1d_3/conv1d/ExpandDims_grad/ReshapeReshapeFgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/control_dependency7gradients/ConvNet/conv1d_3/conv1d/ExpandDims_grad/Shape*
T0*
Tshape0*-
_output_shapes
:џџџџџџџџџќ

9gradients/ConvNet/conv1d_3/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:

;gradients/ConvNet/conv1d_3/conv1d/ExpandDims_1_grad/ReshapeReshapeHgradients/ConvNet/conv1d_3/conv1d/Conv2D_grad/tuple/control_dependency_19gradients/ConvNet/conv1d_3/conv1d/ExpandDims_1_grad/Shape*$
_output_shapes
:*
T0*
Tshape0

4gradients/ConvNet/max_pooling1d_2/Squeeze_grad/ShapeShapeConvNet/max_pooling1d_2/MaxPool*
T0*
out_type0*
_output_shapes
:
ќ
6gradients/ConvNet/max_pooling1d_2/Squeeze_grad/ReshapeReshape9gradients/ConvNet/conv1d_3/conv1d/ExpandDims_grad/Reshape4gradients/ConvNet/max_pooling1d_2/Squeeze_grad/Shape*1
_output_shapes
:џџџџџџџџџќ*
T0*
Tshape0
е
:gradients/ConvNet/max_pooling1d_2/MaxPool_grad/MaxPoolGradMaxPoolGrad"ConvNet/max_pooling1d_2/ExpandDimsConvNet/max_pooling1d_2/MaxPool6gradients/ConvNet/max_pooling1d_2/Squeeze_grad/Reshape*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџљ

7gradients/ConvNet/max_pooling1d_2/ExpandDims_grad/ShapeShapeConvNet/conv1d_2/Relu*
T0*
out_type0*
_output_shapes
:
џ
9gradients/ConvNet/max_pooling1d_2/ExpandDims_grad/ReshapeReshape:gradients/ConvNet/max_pooling1d_2/MaxPool_grad/MaxPoolGrad7gradients/ConvNet/max_pooling1d_2/ExpandDims_grad/Shape*
T0*
Tshape0*-
_output_shapes
:џџџџџџџџџљ
У
-gradients/ConvNet/conv1d_2/Relu_grad/ReluGradReluGrad9gradients/ConvNet/max_pooling1d_2/ExpandDims_grad/ReshapeConvNet/conv1d_2/Relu*-
_output_shapes
:џџџџџџџџџљ*
T0
Ў
3gradients/ConvNet/conv1d_2/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/ConvNet/conv1d_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
І
8gradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/group_depsNoOp4^gradients/ConvNet/conv1d_2/BiasAdd_grad/BiasAddGrad.^gradients/ConvNet/conv1d_2/Relu_grad/ReluGrad
А
@gradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/ConvNet/conv1d_2/Relu_grad/ReluGrad9^gradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/ConvNet/conv1d_2/Relu_grad/ReluGrad*-
_output_shapes
:џџџџџџџџџљ
Ќ
Bgradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/ConvNet/conv1d_2/BiasAdd_grad/BiasAddGrad9^gradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*F
_class<
:8loc:@gradients/ConvNet/conv1d_2/BiasAdd_grad/BiasAddGrad

4gradients/ConvNet/conv1d_2/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d_2/conv1d/Conv2D*
T0*
out_type0*
_output_shapes
:

6gradients/ConvNet/conv1d_2/conv1d/Squeeze_grad/ReshapeReshape@gradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/control_dependency4gradients/ConvNet/conv1d_2/conv1d/Squeeze_grad/Shape*1
_output_shapes
:џџџџџџџџџљ*
T0*
Tshape0
Ь
4gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/ShapeNShapeN"ConvNet/conv1d_2/conv1d/ExpandDims$ConvNet/conv1d_2/conv1d/ExpandDims_1*
T0*
out_type0*
N* 
_output_shapes
::

Agradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput4gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/ShapeN$ConvNet/conv1d_2/conv1d/ExpandDims_16gradients/ConvNet/conv1d_2/conv1d/Squeeze_grad/Reshape*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*1
_output_shapes
:џџџџџџџџџ§

Bgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"ConvNet/conv1d_2/conv1d/ExpandDims6gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/ShapeN:16gradients/ConvNet/conv1d_2/conv1d/Squeeze_grad/Reshape*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*(
_output_shapes
:
Я
>gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/group_depsNoOpC^gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropFilterB^gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropInput
ш
Fgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/control_dependencyIdentityAgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropInput?^gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ§
у
Hgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityBgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropFilter?^gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:

7gradients/ConvNet/conv1d_2/conv1d/ExpandDims_grad/ShapeShapeConvNet/max_pooling1d_1/Squeeze*
_output_shapes
:*
T0*
out_type0

9gradients/ConvNet/conv1d_2/conv1d/ExpandDims_grad/ReshapeReshapeFgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/control_dependency7gradients/ConvNet/conv1d_2/conv1d/ExpandDims_grad/Shape*
Tshape0*-
_output_shapes
:џџџџџџџџџ§*
T0

9gradients/ConvNet/conv1d_2/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:

;gradients/ConvNet/conv1d_2/conv1d/ExpandDims_1_grad/ReshapeReshapeHgradients/ConvNet/conv1d_2/conv1d/Conv2D_grad/tuple/control_dependency_19gradients/ConvNet/conv1d_2/conv1d/ExpandDims_1_grad/Shape*$
_output_shapes
:*
T0*
Tshape0

4gradients/ConvNet/max_pooling1d_1/Squeeze_grad/ShapeShapeConvNet/max_pooling1d_1/MaxPool*
T0*
out_type0*
_output_shapes
:
ќ
6gradients/ConvNet/max_pooling1d_1/Squeeze_grad/ReshapeReshape9gradients/ConvNet/conv1d_2/conv1d/ExpandDims_grad/Reshape4gradients/ConvNet/max_pooling1d_1/Squeeze_grad/Shape*
T0*
Tshape0*1
_output_shapes
:џџџџџџџџџ§
е
:gradients/ConvNet/max_pooling1d_1/MaxPool_grad/MaxPoolGradMaxPoolGrad"ConvNet/max_pooling1d_1/ExpandDimsConvNet/max_pooling1d_1/MaxPool6gradients/ConvNet/max_pooling1d_1/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџњ*
T0

7gradients/ConvNet/max_pooling1d_1/ExpandDims_grad/ShapeShapeConvNet/conv1d_1/Relu*
_output_shapes
:*
T0*
out_type0
џ
9gradients/ConvNet/max_pooling1d_1/ExpandDims_grad/ReshapeReshape:gradients/ConvNet/max_pooling1d_1/MaxPool_grad/MaxPoolGrad7gradients/ConvNet/max_pooling1d_1/ExpandDims_grad/Shape*
Tshape0*-
_output_shapes
:џџџџџџџџџњ*
T0
У
-gradients/ConvNet/conv1d_1/Relu_grad/ReluGradReluGrad9gradients/ConvNet/max_pooling1d_1/ExpandDims_grad/ReshapeConvNet/conv1d_1/Relu*
T0*-
_output_shapes
:џџџџџџџџџњ
Ў
3gradients/ConvNet/conv1d_1/BiasAdd_grad/BiasAddGradBiasAddGrad-gradients/ConvNet/conv1d_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
І
8gradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients/ConvNet/conv1d_1/BiasAdd_grad/BiasAddGrad.^gradients/ConvNet/conv1d_1/Relu_grad/ReluGrad
А
@gradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/control_dependencyIdentity-gradients/ConvNet/conv1d_1/Relu_grad/ReluGrad9^gradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/ConvNet/conv1d_1/Relu_grad/ReluGrad*-
_output_shapes
:џџџџџџџџџњ
Ќ
Bgradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/control_dependency_1Identity3gradients/ConvNet/conv1d_1/BiasAdd_grad/BiasAddGrad9^gradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*F
_class<
:8loc:@gradients/ConvNet/conv1d_1/BiasAdd_grad/BiasAddGrad

4gradients/ConvNet/conv1d_1/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d_1/conv1d/Conv2D*
T0*
out_type0*
_output_shapes
:

6gradients/ConvNet/conv1d_1/conv1d/Squeeze_grad/ReshapeReshape@gradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/control_dependency4gradients/ConvNet/conv1d_1/conv1d/Squeeze_grad/Shape*
Tshape0*1
_output_shapes
:џџџџџџџџџњ*
T0
Ь
4gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/ShapeNShapeN"ConvNet/conv1d_1/conv1d/ExpandDims$ConvNet/conv1d_1/conv1d/ExpandDims_1*
T0*
out_type0*
N* 
_output_shapes
::

Agradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput4gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/ShapeN$ConvNet/conv1d_1/conv1d/ExpandDims_16gradients/ConvNet/conv1d_1/conv1d/Squeeze_grad/Reshape*
paddingVALID*1
_output_shapes
:џџџџџџџџџў*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

Bgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter"ConvNet/conv1d_1/conv1d/ExpandDims6gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/ShapeN:16gradients/ConvNet/conv1d_1/conv1d/Squeeze_grad/Reshape*(
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
Я
>gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/group_depsNoOpC^gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropFilterB^gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropInput
ш
Fgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/control_dependencyIdentityAgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropInput?^gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџў
у
Hgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/control_dependency_1IdentityBgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropFilter?^gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/Conv2DBackpropFilter*(
_output_shapes
:

7gradients/ConvNet/conv1d_1/conv1d/ExpandDims_grad/ShapeShapeConvNet/max_pooling1d/Squeeze*
out_type0*
_output_shapes
:*
T0

9gradients/ConvNet/conv1d_1/conv1d/ExpandDims_grad/ReshapeReshapeFgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/control_dependency7gradients/ConvNet/conv1d_1/conv1d/ExpandDims_grad/Shape*-
_output_shapes
:џџџџџџџџџў*
T0*
Tshape0

9gradients/ConvNet/conv1d_1/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:

;gradients/ConvNet/conv1d_1/conv1d/ExpandDims_1_grad/ReshapeReshapeHgradients/ConvNet/conv1d_1/conv1d/Conv2D_grad/tuple/control_dependency_19gradients/ConvNet/conv1d_1/conv1d/ExpandDims_1_grad/Shape*
T0*
Tshape0*$
_output_shapes
:

2gradients/ConvNet/max_pooling1d/Squeeze_grad/ShapeShapeConvNet/max_pooling1d/MaxPool*
out_type0*
_output_shapes
:*
T0
ј
4gradients/ConvNet/max_pooling1d/Squeeze_grad/ReshapeReshape9gradients/ConvNet/conv1d_1/conv1d/ExpandDims_grad/Reshape2gradients/ConvNet/max_pooling1d/Squeeze_grad/Shape*
Tshape0*1
_output_shapes
:џџџџџџџџџў*
T0
Э
8gradients/ConvNet/max_pooling1d/MaxPool_grad/MaxPoolGradMaxPoolGrad ConvNet/max_pooling1d/ExpandDimsConvNet/max_pooling1d/MaxPool4gradients/ConvNet/max_pooling1d/Squeeze_grad/Reshape*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*1
_output_shapes
:џџџџџџџџџќ*
T0

5gradients/ConvNet/max_pooling1d/ExpandDims_grad/ShapeShapeConvNet/conv1d/Relu*
T0*
out_type0*
_output_shapes
:
љ
7gradients/ConvNet/max_pooling1d/ExpandDims_grad/ReshapeReshape8gradients/ConvNet/max_pooling1d/MaxPool_grad/MaxPoolGrad5gradients/ConvNet/max_pooling1d/ExpandDims_grad/Shape*
T0*
Tshape0*-
_output_shapes
:џџџџџџџџџќ
Н
+gradients/ConvNet/conv1d/Relu_grad/ReluGradReluGrad7gradients/ConvNet/max_pooling1d/ExpandDims_grad/ReshapeConvNet/conv1d/Relu*-
_output_shapes
:џџџџџџџџџќ*
T0
Њ
1gradients/ConvNet/conv1d/BiasAdd_grad/BiasAddGradBiasAddGrad+gradients/ConvNet/conv1d/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
 
6gradients/ConvNet/conv1d/BiasAdd_grad/tuple/group_depsNoOp2^gradients/ConvNet/conv1d/BiasAdd_grad/BiasAddGrad,^gradients/ConvNet/conv1d/Relu_grad/ReluGrad
Ј
>gradients/ConvNet/conv1d/BiasAdd_grad/tuple/control_dependencyIdentity+gradients/ConvNet/conv1d/Relu_grad/ReluGrad7^gradients/ConvNet/conv1d/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/ConvNet/conv1d/Relu_grad/ReluGrad*-
_output_shapes
:џџџџџџџџџќ
Є
@gradients/ConvNet/conv1d/BiasAdd_grad/tuple/control_dependency_1Identity1gradients/ConvNet/conv1d/BiasAdd_grad/BiasAddGrad7^gradients/ConvNet/conv1d/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/ConvNet/conv1d/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

2gradients/ConvNet/conv1d/conv1d/Squeeze_grad/ShapeShapeConvNet/conv1d/conv1d/Conv2D*
T0*
out_type0*
_output_shapes
:
§
4gradients/ConvNet/conv1d/conv1d/Squeeze_grad/ReshapeReshape>gradients/ConvNet/conv1d/BiasAdd_grad/tuple/control_dependency2gradients/ConvNet/conv1d/conv1d/Squeeze_grad/Shape*1
_output_shapes
:џџџџџџџџџќ*
T0*
Tshape0
Ц
2gradients/ConvNet/conv1d/conv1d/Conv2D_grad/ShapeNShapeN ConvNet/conv1d/conv1d/ExpandDims"ConvNet/conv1d/conv1d/ExpandDims_1*
out_type0*
N* 
_output_shapes
::*
T0

?gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput2gradients/ConvNet/conv1d/conv1d/Conv2D_grad/ShapeN"ConvNet/conv1d/conv1d/ExpandDims_14gradients/ConvNet/conv1d/conv1d/Squeeze_grad/Reshape*0
_output_shapes
:џџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID

@gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter ConvNet/conv1d/conv1d/ExpandDims4gradients/ConvNet/conv1d/conv1d/Conv2D_grad/ShapeN:14gradients/ConvNet/conv1d/conv1d/Squeeze_grad/Reshape*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:*
	dilations
*
T0
Щ
<gradients/ConvNet/conv1d/conv1d/Conv2D_grad/tuple/group_depsNoOpA^gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropFilter@^gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropInput
п
Dgradients/ConvNet/conv1d/conv1d/Conv2D_grad/tuple/control_dependencyIdentity?gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropInput=^gradients/ConvNet/conv1d/conv1d/Conv2D_grad/tuple/group_deps*0
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropInput
к
Fgradients/ConvNet/conv1d/conv1d/Conv2D_grad/tuple/control_dependency_1Identity@gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropFilter=^gradients/ConvNet/conv1d/conv1d/Conv2D_grad/tuple/group_deps*'
_output_shapes
:*
T0*S
_classI
GEloc:@gradients/ConvNet/conv1d/conv1d/Conv2D_grad/Conv2DBackpropFilter

7gradients/ConvNet/conv1d/conv1d/ExpandDims_1_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:

9gradients/ConvNet/conv1d/conv1d/ExpandDims_1_grad/ReshapeReshapeFgradients/ConvNet/conv1d/conv1d/Conv2D_grad/tuple/control_dependency_17gradients/ConvNet/conv1d/conv1d/ExpandDims_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:

beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*.
_class$
" loc:@ConvNet/cnn_logits_out/bias

beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape: 
О
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
z
beta1_power/readIdentitybeta1_power*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
: 

beta2_power/initial_valueConst*
valueB
 *wО?*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias
О
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
z
beta2_power/readIdentitybeta2_power*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
: *
T0
Л
<ConvNet/conv1d/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*(
_class
loc:@ConvNet/conv1d/kernel*!
valueB"         *
dtype0
Ё
2ConvNet/conv1d/kernel/Adam/Initializer/zeros/ConstConst*(
_class
loc:@ConvNet/conv1d/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

,ConvNet/conv1d/kernel/Adam/Initializer/zerosFill<ConvNet/conv1d/kernel/Adam/Initializer/zeros/shape_as_tensor2ConvNet/conv1d/kernel/Adam/Initializer/zeros/Const*#
_output_shapes
:*
T0*(
_class
loc:@ConvNet/conv1d/kernel*

index_type0
Т
ConvNet/conv1d/kernel/Adam
VariableV2*
shape:*
dtype0*#
_output_shapes
:*
shared_name *(
_class
loc:@ConvNet/conv1d/kernel*
	container 
і
!ConvNet/conv1d/kernel/Adam/AssignAssignConvNet/conv1d/kernel/Adam,ConvNet/conv1d/kernel/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d/kernel*
validate_shape(*#
_output_shapes
:

ConvNet/conv1d/kernel/Adam/readIdentityConvNet/conv1d/kernel/Adam*
T0*(
_class
loc:@ConvNet/conv1d/kernel*#
_output_shapes
:
Н
>ConvNet/conv1d/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*(
_class
loc:@ConvNet/conv1d/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ѓ
4ConvNet/conv1d/kernel/Adam_1/Initializer/zeros/ConstConst*(
_class
loc:@ConvNet/conv1d/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.ConvNet/conv1d/kernel/Adam_1/Initializer/zerosFill>ConvNet/conv1d/kernel/Adam_1/Initializer/zeros/shape_as_tensor4ConvNet/conv1d/kernel/Adam_1/Initializer/zeros/Const*#
_output_shapes
:*
T0*(
_class
loc:@ConvNet/conv1d/kernel*

index_type0
Ф
ConvNet/conv1d/kernel/Adam_1
VariableV2*(
_class
loc:@ConvNet/conv1d/kernel*
	container *
shape:*
dtype0*#
_output_shapes
:*
shared_name 
ќ
#ConvNet/conv1d/kernel/Adam_1/AssignAssignConvNet/conv1d/kernel/Adam_1.ConvNet/conv1d/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d/kernel*
validate_shape(*#
_output_shapes
:
Ѓ
!ConvNet/conv1d/kernel/Adam_1/readIdentityConvNet/conv1d/kernel/Adam_1*
T0*(
_class
loc:@ConvNet/conv1d/kernel*#
_output_shapes
:
Ё
*ConvNet/conv1d/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*&
_class
loc:@ConvNet/conv1d/bias*
valueB*    *
dtype0
Ў
ConvNet/conv1d/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *&
_class
loc:@ConvNet/conv1d/bias*
	container *
shape:
ц
ConvNet/conv1d/bias/Adam/AssignAssignConvNet/conv1d/bias/Adam*ConvNet/conv1d/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*&
_class
loc:@ConvNet/conv1d/bias*
validate_shape(

ConvNet/conv1d/bias/Adam/readIdentityConvNet/conv1d/bias/Adam*
T0*&
_class
loc:@ConvNet/conv1d/bias*
_output_shapes	
:
Ѓ
,ConvNet/conv1d/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*&
_class
loc:@ConvNet/conv1d/bias*
valueB*    *
dtype0
А
ConvNet/conv1d/bias/Adam_1
VariableV2*&
_class
loc:@ConvNet/conv1d/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ь
!ConvNet/conv1d/bias/Adam_1/AssignAssignConvNet/conv1d/bias/Adam_1,ConvNet/conv1d/bias/Adam_1/Initializer/zeros*&
_class
loc:@ConvNet/conv1d/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

ConvNet/conv1d/bias/Adam_1/readIdentityConvNet/conv1d/bias/Adam_1*&
_class
loc:@ConvNet/conv1d/bias*
_output_shapes	
:*
T0
П
>ConvNet/conv1d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_1/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ѕ
4ConvNet/conv1d_1/kernel/Adam/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.ConvNet/conv1d_1/kernel/Adam/Initializer/zerosFill>ConvNet/conv1d_1/kernel/Adam/Initializer/zeros/shape_as_tensor4ConvNet/conv1d_1/kernel/Adam/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*

index_type0*$
_output_shapes
:
Ш
ConvNet/conv1d_1/kernel/Adam
VariableV2**
_class 
loc:@ConvNet/conv1d_1/kernel*
	container *
shape:*
dtype0*$
_output_shapes
:*
shared_name 
џ
#ConvNet/conv1d_1/kernel/Adam/AssignAssignConvNet/conv1d_1/kernel/Adam.ConvNet/conv1d_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*
validate_shape(*$
_output_shapes
:
І
!ConvNet/conv1d_1/kernel/Adam/readIdentityConvNet/conv1d_1/kernel/Adam*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel
С
@ConvNet/conv1d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_1/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: **
_class 
loc:@ConvNet/conv1d_1/kernel*
valueB
 *    

0ConvNet/conv1d_1/kernel/Adam_1/Initializer/zerosFill@ConvNet/conv1d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor6ConvNet/conv1d_1/kernel/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*

index_type0*$
_output_shapes
:
Ъ
ConvNet/conv1d_1/kernel/Adam_1
VariableV2*
	container *
shape:*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_1/kernel

%ConvNet/conv1d_1/kernel/Adam_1/AssignAssignConvNet/conv1d_1/kernel/Adam_10ConvNet/conv1d_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel
Њ
#ConvNet/conv1d_1/kernel/Adam_1/readIdentityConvNet/conv1d_1/kernel/Adam_1*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel
Ѕ
,ConvNet/conv1d_1/bias/Adam/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
В
ConvNet/conv1d_1/bias/Adam
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ю
!ConvNet/conv1d_1/bias/Adam/AssignAssignConvNet/conv1d_1/bias/Adam,ConvNet/conv1d_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_1/bias*
validate_shape(*
_output_shapes	
:

ConvNet/conv1d_1/bias/Adam/readIdentityConvNet/conv1d_1/bias/Adam*(
_class
loc:@ConvNet/conv1d_1/bias*
_output_shapes	
:*
T0
Ї
.ConvNet/conv1d_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*(
_class
loc:@ConvNet/conv1d_1/bias*
valueB*    
Д
ConvNet/conv1d_1/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_1/bias
є
#ConvNet/conv1d_1/bias/Adam_1/AssignAssignConvNet/conv1d_1/bias/Adam_1.ConvNet/conv1d_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_1/bias*
validate_shape(*
_output_shapes	
:

!ConvNet/conv1d_1/bias/Adam_1/readIdentityConvNet/conv1d_1/bias/Adam_1*
_output_shapes	
:*
T0*(
_class
loc:@ConvNet/conv1d_1/bias
П
>ConvNet/conv1d_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:**
_class 
loc:@ConvNet/conv1d_2/kernel*!
valueB"         
Ѕ
4ConvNet/conv1d_2/kernel/Adam/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.ConvNet/conv1d_2/kernel/Adam/Initializer/zerosFill>ConvNet/conv1d_2/kernel/Adam/Initializer/zeros/shape_as_tensor4ConvNet/conv1d_2/kernel/Adam/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*

index_type0*$
_output_shapes
:
Ш
ConvNet/conv1d_2/kernel/Adam
VariableV2*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_2/kernel*
	container *
shape:*
dtype0
џ
#ConvNet/conv1d_2/kernel/Adam/AssignAssignConvNet/conv1d_2/kernel/Adam.ConvNet/conv1d_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
validate_shape(*$
_output_shapes
:
І
!ConvNet/conv1d_2/kernel/Adam/readIdentityConvNet/conv1d_2/kernel/Adam*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel
С
@ConvNet/conv1d_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_2/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_2/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: **
_class 
loc:@ConvNet/conv1d_2/kernel*
valueB
 *    *
dtype0

0ConvNet/conv1d_2/kernel/Adam_1/Initializer/zerosFill@ConvNet/conv1d_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor6ConvNet/conv1d_2/kernel/Adam_1/Initializer/zeros/Const**
_class 
loc:@ConvNet/conv1d_2/kernel*

index_type0*$
_output_shapes
:*
T0
Ъ
ConvNet/conv1d_2/kernel/Adam_1
VariableV2*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_2/kernel*
	container *
shape:

%ConvNet/conv1d_2/kernel/Adam_1/AssignAssignConvNet/conv1d_2/kernel/Adam_10ConvNet/conv1d_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
validate_shape(*$
_output_shapes
:
Њ
#ConvNet/conv1d_2/kernel/Adam_1/readIdentityConvNet/conv1d_2/kernel/Adam_1*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*$
_output_shapes
:
Ѕ
,ConvNet/conv1d_2/bias/Adam/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
В
ConvNet/conv1d_2/bias/Adam
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ю
!ConvNet/conv1d_2/bias/Adam/AssignAssignConvNet/conv1d_2/bias/Adam,ConvNet/conv1d_2/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
validate_shape(

ConvNet/conv1d_2/bias/Adam/readIdentityConvNet/conv1d_2/bias/Adam*
_output_shapes	
:*
T0*(
_class
loc:@ConvNet/conv1d_2/bias
Ї
.ConvNet/conv1d_2/bias/Adam_1/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
Д
ConvNet/conv1d_2/bias/Adam_1
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
є
#ConvNet/conv1d_2/bias/Adam_1/AssignAssignConvNet/conv1d_2/bias/Adam_1.ConvNet/conv1d_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
validate_shape(*
_output_shapes	
:

!ConvNet/conv1d_2/bias/Adam_1/readIdentityConvNet/conv1d_2/bias/Adam_1*
_output_shapes	
:*
T0*(
_class
loc:@ConvNet/conv1d_2/bias
П
>ConvNet/conv1d_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_3/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ѕ
4ConvNet/conv1d_3/kernel/Adam/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.ConvNet/conv1d_3/kernel/Adam/Initializer/zerosFill>ConvNet/conv1d_3/kernel/Adam/Initializer/zeros/shape_as_tensor4ConvNet/conv1d_3/kernel/Adam/Initializer/zeros/Const*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*

index_type0
Ш
ConvNet/conv1d_3/kernel/Adam
VariableV2*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_3/kernel*
	container *
shape:*
dtype0
џ
#ConvNet/conv1d_3/kernel/Adam/AssignAssignConvNet/conv1d_3/kernel/Adam.ConvNet/conv1d_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
validate_shape(*$
_output_shapes
:
І
!ConvNet/conv1d_3/kernel/Adam/readIdentityConvNet/conv1d_3/kernel/Adam*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*$
_output_shapes
:
С
@ConvNet/conv1d_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:**
_class 
loc:@ConvNet/conv1d_3/kernel*!
valueB"         
Ї
6ConvNet/conv1d_3/kernel/Adam_1/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0ConvNet/conv1d_3/kernel/Adam_1/Initializer/zerosFill@ConvNet/conv1d_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor6ConvNet/conv1d_3/kernel/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*

index_type0*$
_output_shapes
:
Ъ
ConvNet/conv1d_3/kernel/Adam_1
VariableV2*
	container *
shape:*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_3/kernel

%ConvNet/conv1d_3/kernel/Adam_1/AssignAssignConvNet/conv1d_3/kernel/Adam_10ConvNet/conv1d_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
validate_shape(*$
_output_shapes
:
Њ
#ConvNet/conv1d_3/kernel/Adam_1/readIdentityConvNet/conv1d_3/kernel/Adam_1*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*$
_output_shapes
:
Ѕ
,ConvNet/conv1d_3/bias/Adam/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
В
ConvNet/conv1d_3/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_3/bias
ю
!ConvNet/conv1d_3/bias/Adam/AssignAssignConvNet/conv1d_3/bias/Adam,ConvNet/conv1d_3/bias/Adam/Initializer/zeros*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

ConvNet/conv1d_3/bias/Adam/readIdentityConvNet/conv1d_3/bias/Adam*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
_output_shapes	
:
Ї
.ConvNet/conv1d_3/bias/Adam_1/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Д
ConvNet/conv1d_3/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_3/bias
є
#ConvNet/conv1d_3/bias/Adam_1/AssignAssignConvNet/conv1d_3/bias/Adam_1.ConvNet/conv1d_3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
validate_shape(*
_output_shapes	
:

!ConvNet/conv1d_3/bias/Adam_1/readIdentityConvNet/conv1d_3/bias/Adam_1*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
_output_shapes	
:
П
>ConvNet/conv1d_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:**
_class 
loc:@ConvNet/conv1d_4/kernel*!
valueB"         *
dtype0
Ѕ
4ConvNet/conv1d_4/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: **
_class 
loc:@ConvNet/conv1d_4/kernel*
valueB
 *    *
dtype0

.ConvNet/conv1d_4/kernel/Adam/Initializer/zerosFill>ConvNet/conv1d_4/kernel/Adam/Initializer/zeros/shape_as_tensor4ConvNet/conv1d_4/kernel/Adam/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*

index_type0*$
_output_shapes
:
Ш
ConvNet/conv1d_4/kernel/Adam
VariableV2*
	container *
shape:*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_4/kernel
џ
#ConvNet/conv1d_4/kernel/Adam/AssignAssignConvNet/conv1d_4/kernel/Adam.ConvNet/conv1d_4/kernel/Adam/Initializer/zeros*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel
І
!ConvNet/conv1d_4/kernel/Adam/readIdentityConvNet/conv1d_4/kernel/Adam*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*$
_output_shapes
:
С
@ConvNet/conv1d_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:**
_class 
loc:@ConvNet/conv1d_4/kernel*!
valueB"         
Ї
6ConvNet/conv1d_4/kernel/Adam_1/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0ConvNet/conv1d_4/kernel/Adam_1/Initializer/zerosFill@ConvNet/conv1d_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor6ConvNet/conv1d_4/kernel/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*

index_type0*$
_output_shapes
:
Ъ
ConvNet/conv1d_4/kernel/Adam_1
VariableV2*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_4/kernel*
	container *
shape:

%ConvNet/conv1d_4/kernel/Adam_1/AssignAssignConvNet/conv1d_4/kernel/Adam_10ConvNet/conv1d_4/kernel/Adam_1/Initializer/zeros*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(
Њ
#ConvNet/conv1d_4/kernel/Adam_1/readIdentityConvNet/conv1d_4/kernel/Adam_1**
_class 
loc:@ConvNet/conv1d_4/kernel*$
_output_shapes
:*
T0
Ѕ
,ConvNet/conv1d_4/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*(
_class
loc:@ConvNet/conv1d_4/bias*
valueB*    
В
ConvNet/conv1d_4/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_4/bias
ю
!ConvNet/conv1d_4/bias/Adam/AssignAssignConvNet/conv1d_4/bias/Adam,ConvNet/conv1d_4/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_4/bias

ConvNet/conv1d_4/bias/Adam/readIdentityConvNet/conv1d_4/bias/Adam*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
_output_shapes	
:
Ї
.ConvNet/conv1d_4/bias/Adam_1/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_4/bias*
valueB*    *
dtype0*
_output_shapes	
:
Д
ConvNet/conv1d_4/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *(
_class
loc:@ConvNet/conv1d_4/bias*
	container *
shape:
є
#ConvNet/conv1d_4/bias/Adam_1/AssignAssignConvNet/conv1d_4/bias/Adam_1.ConvNet/conv1d_4/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_4/bias

!ConvNet/conv1d_4/bias/Adam_1/readIdentityConvNet/conv1d_4/bias/Adam_1*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
_output_shapes	
:
П
>ConvNet/conv1d_5/kernel/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_5/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ѕ
4ConvNet/conv1d_5/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: **
_class 
loc:@ConvNet/conv1d_5/kernel*
valueB
 *    

.ConvNet/conv1d_5/kernel/Adam/Initializer/zerosFill>ConvNet/conv1d_5/kernel/Adam/Initializer/zeros/shape_as_tensor4ConvNet/conv1d_5/kernel/Adam/Initializer/zeros/Const*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*

index_type0
Ш
ConvNet/conv1d_5/kernel/Adam
VariableV2**
_class 
loc:@ConvNet/conv1d_5/kernel*
	container *
shape:*
dtype0*$
_output_shapes
:*
shared_name 
џ
#ConvNet/conv1d_5/kernel/Adam/AssignAssignConvNet/conv1d_5/kernel/Adam.ConvNet/conv1d_5/kernel/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*
validate_shape(*$
_output_shapes
:
І
!ConvNet/conv1d_5/kernel/Adam/readIdentityConvNet/conv1d_5/kernel/Adam*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*$
_output_shapes
:
С
@ConvNet/conv1d_5/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_5/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ї
6ConvNet/conv1d_5/kernel/Adam_1/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0ConvNet/conv1d_5/kernel/Adam_1/Initializer/zerosFill@ConvNet/conv1d_5/kernel/Adam_1/Initializer/zeros/shape_as_tensor6ConvNet/conv1d_5/kernel/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*

index_type0*$
_output_shapes
:
Ъ
ConvNet/conv1d_5/kernel/Adam_1
VariableV2*
shared_name **
_class 
loc:@ConvNet/conv1d_5/kernel*
	container *
shape:*
dtype0*$
_output_shapes
:

%ConvNet/conv1d_5/kernel/Adam_1/AssignAssignConvNet/conv1d_5/kernel/Adam_10ConvNet/conv1d_5/kernel/Adam_1/Initializer/zeros**
_class 
loc:@ConvNet/conv1d_5/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0
Њ
#ConvNet/conv1d_5/kernel/Adam_1/readIdentityConvNet/conv1d_5/kernel/Adam_1**
_class 
loc:@ConvNet/conv1d_5/kernel*$
_output_shapes
:*
T0
Ѕ
,ConvNet/conv1d_5/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*(
_class
loc:@ConvNet/conv1d_5/bias*
valueB*    
В
ConvNet/conv1d_5/bias/Adam
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_5/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ю
!ConvNet/conv1d_5/bias/Adam/AssignAssignConvNet/conv1d_5/bias/Adam,ConvNet/conv1d_5/bias/Adam/Initializer/zeros*(
_class
loc:@ConvNet/conv1d_5/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

ConvNet/conv1d_5/bias/Adam/readIdentityConvNet/conv1d_5/bias/Adam*
T0*(
_class
loc:@ConvNet/conv1d_5/bias*
_output_shapes	
:
Ї
.ConvNet/conv1d_5/bias/Adam_1/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_5/bias*
valueB*    *
dtype0*
_output_shapes	
:
Д
ConvNet/conv1d_5/bias/Adam_1
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_5/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
є
#ConvNet/conv1d_5/bias/Adam_1/AssignAssignConvNet/conv1d_5/bias/Adam_1.ConvNet/conv1d_5/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_5/bias*
validate_shape(*
_output_shapes	
:

!ConvNet/conv1d_5/bias/Adam_1/readIdentityConvNet/conv1d_5/bias/Adam_1*
_output_shapes	
:*
T0*(
_class
loc:@ConvNet/conv1d_5/bias
П
>ConvNet/conv1d_6/kernel/Adam/Initializer/zeros/shape_as_tensorConst**
_class 
loc:@ConvNet/conv1d_6/kernel*!
valueB"         *
dtype0*
_output_shapes
:
Ѕ
4ConvNet/conv1d_6/kernel/Adam/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_6/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

.ConvNet/conv1d_6/kernel/Adam/Initializer/zerosFill>ConvNet/conv1d_6/kernel/Adam/Initializer/zeros/shape_as_tensor4ConvNet/conv1d_6/kernel/Adam/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*

index_type0*$
_output_shapes
:
Ш
ConvNet/conv1d_6/kernel/Adam
VariableV2*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_6/kernel*
	container *
shape:*
dtype0
џ
#ConvNet/conv1d_6/kernel/Adam/AssignAssignConvNet/conv1d_6/kernel/Adam.ConvNet/conv1d_6/kernel/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
validate_shape(*$
_output_shapes
:
І
!ConvNet/conv1d_6/kernel/Adam/readIdentityConvNet/conv1d_6/kernel/Adam*$
_output_shapes
:*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel
С
@ConvNet/conv1d_6/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:**
_class 
loc:@ConvNet/conv1d_6/kernel*!
valueB"         *
dtype0
Ї
6ConvNet/conv1d_6/kernel/Adam_1/Initializer/zeros/ConstConst**
_class 
loc:@ConvNet/conv1d_6/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

0ConvNet/conv1d_6/kernel/Adam_1/Initializer/zerosFill@ConvNet/conv1d_6/kernel/Adam_1/Initializer/zeros/shape_as_tensor6ConvNet/conv1d_6/kernel/Adam_1/Initializer/zeros/Const*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*

index_type0*$
_output_shapes
:
Ъ
ConvNet/conv1d_6/kernel/Adam_1
VariableV2*
shape:*
dtype0*$
_output_shapes
:*
shared_name **
_class 
loc:@ConvNet/conv1d_6/kernel*
	container 

%ConvNet/conv1d_6/kernel/Adam_1/AssignAssignConvNet/conv1d_6/kernel/Adam_10ConvNet/conv1d_6/kernel/Adam_1/Initializer/zeros*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(
Њ
#ConvNet/conv1d_6/kernel/Adam_1/readIdentityConvNet/conv1d_6/kernel/Adam_1*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*$
_output_shapes
:
Ѕ
,ConvNet/conv1d_6/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*(
_class
loc:@ConvNet/conv1d_6/bias*
valueB*    
В
ConvNet/conv1d_6/bias/Adam
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_6/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ю
!ConvNet/conv1d_6/bias/Adam/AssignAssignConvNet/conv1d_6/bias/Adam,ConvNet/conv1d_6/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_6/bias

ConvNet/conv1d_6/bias/Adam/readIdentityConvNet/conv1d_6/bias/Adam*(
_class
loc:@ConvNet/conv1d_6/bias*
_output_shapes	
:*
T0
Ї
.ConvNet/conv1d_6/bias/Adam_1/Initializer/zerosConst*(
_class
loc:@ConvNet/conv1d_6/bias*
valueB*    *
dtype0*
_output_shapes	
:
Д
ConvNet/conv1d_6/bias/Adam_1
VariableV2*
shared_name *(
_class
loc:@ConvNet/conv1d_6/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
є
#ConvNet/conv1d_6/bias/Adam_1/AssignAssignConvNet/conv1d_6/bias/Adam_1.ConvNet/conv1d_6/bias/Adam_1/Initializer/zeros*
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

!ConvNet/conv1d_6/bias/Adam_1/readIdentityConvNet/conv1d_6/bias/Adam_1*
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
_output_shapes	
:
Е
;ConvNet/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@ConvNet/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

1ConvNet/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@ConvNet/dense/kernel*
valueB
 *    

+ConvNet/dense/kernel/Adam/Initializer/zerosFill;ConvNet/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1ConvNet/dense/kernel/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@ConvNet/dense/kernel*

index_type0* 
_output_shapes
:

К
ConvNet/dense/kernel/Adam
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

я
 ConvNet/dense/kernel/Adam/AssignAssignConvNet/dense/kernel/Adam+ConvNet/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:


ConvNet/dense/kernel/Adam/readIdentityConvNet/dense/kernel/Adam* 
_output_shapes
:
*
T0*'
_class
loc:@ConvNet/dense/kernel
З
=ConvNet/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@ConvNet/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ё
3ConvNet/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *'
_class
loc:@ConvNet/dense/kernel*
valueB
 *    *
dtype0

-ConvNet/dense/kernel/Adam_1/Initializer/zerosFill=ConvNet/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3ConvNet/dense/kernel/Adam_1/Initializer/zeros/Const*'
_class
loc:@ConvNet/dense/kernel*

index_type0* 
_output_shapes
:
*
T0
М
ConvNet/dense/kernel/Adam_1
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ѕ
"ConvNet/dense/kernel/Adam_1/AssignAssignConvNet/dense/kernel/Adam_1-ConvNet/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense/kernel/Adam_1/readIdentityConvNet/dense/kernel/Adam_1*
T0*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:


)ConvNet/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@ConvNet/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ќ
ConvNet/dense/bias/Adam
VariableV2*
shared_name *%
_class
loc:@ConvNet/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
т
ConvNet/dense/bias/Adam/AssignAssignConvNet/dense/bias/Adam)ConvNet/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias

ConvNet/dense/bias/Adam/readIdentityConvNet/dense/bias/Adam*
T0*%
_class
loc:@ConvNet/dense/bias*
_output_shapes	
:
Ё
+ConvNet/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*%
_class
loc:@ConvNet/dense/bias*
valueB*    *
dtype0
Ў
ConvNet/dense/bias/Adam_1
VariableV2*
shared_name *%
_class
loc:@ConvNet/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ш
 ConvNet/dense/bias/Adam_1/AssignAssignConvNet/dense/bias/Adam_1+ConvNet/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(

ConvNet/dense/bias/Adam_1/readIdentityConvNet/dense/bias/Adam_1*
T0*%
_class
loc:@ConvNet/dense/bias*
_output_shapes	
:
Й
=ConvNet/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ѓ
3ConvNet/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@ConvNet/dense_1/kernel*
valueB
 *    

-ConvNet/dense_1/kernel/Adam/Initializer/zerosFill=ConvNet/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_1/kernel*

index_type0* 
_output_shapes
:

О
ConvNet/dense_1/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_1/kernel*
	container 
ї
"ConvNet/dense_1/kernel/Adam/AssignAssignConvNet/dense_1/kernel/Adam-ConvNet/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(

 ConvNet/dense_1/kernel/Adam/readIdentityConvNet/dense_1/kernel/Adam* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_1/kernel
Л
?ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ѕ
5ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_1/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_1/kernel*

index_type0* 
_output_shapes
:

Р
ConvNet/dense_1/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_1/kernel*
	container *
shape:
*
dtype0
§
$ConvNet/dense_1/kernel/Adam_1/AssignAssignConvNet/dense_1/kernel/Adam_1/ConvNet/dense_1/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(
Ѓ
"ConvNet/dense_1/kernel/Adam_1/readIdentityConvNet/dense_1/kernel/Adam_1*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:

Ѓ
+ConvNet/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
А
ConvNet/dense_1/bias/Adam
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ъ
 ConvNet/dense_1/bias/Adam/AssignAssignConvNet/dense_1/bias/Adam+ConvNet/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_1/bias/Adam/readIdentityConvNet/dense_1/bias/Adam*
T0*'
_class
loc:@ConvNet/dense_1/bias*
_output_shapes	
:
Ѕ
-ConvNet/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
В
ConvNet/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_1/bias*
	container *
shape:
№
"ConvNet/dense_1/bias/Adam_1/AssignAssignConvNet/dense_1/bias/Adam_1-ConvNet/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_1/bias/Adam_1/readIdentityConvNet/dense_1/bias/Adam_1*
T0*'
_class
loc:@ConvNet/dense_1/bias*
_output_shapes	
:
Ч
DConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB"      *
dtype0*
_output_shapes
:
Б
:ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
4ConvNet/cnn_logits_out/kernel/Adam/Initializer/zerosFillDConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/shape_as_tensor:ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/Const*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*

index_type0*
_output_shapes
:	
Ъ
"ConvNet/cnn_logits_out/kernel/Adam
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
	container 

)ConvNet/cnn_logits_out/kernel/Adam/AssignAssign"ConvNet/cnn_logits_out/kernel/Adam4ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Г
'ConvNet/cnn_logits_out/kernel/Adam/readIdentity"ConvNet/cnn_logits_out/kernel/Adam*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	*
T0
Щ
FConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB"      
Г
<ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
В
6ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zerosFillFConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/shape_as_tensor<ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*

index_type0*
_output_shapes
:	
Ь
$ConvNet/cnn_logits_out/kernel/Adam_1
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel

+ConvNet/cnn_logits_out/kernel/Adam_1/AssignAssign$ConvNet/cnn_logits_out/kernel/Adam_16ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
З
)ConvNet/cnn_logits_out/kernel/Adam_1/readIdentity$ConvNet/cnn_logits_out/kernel/Adam_1*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	
Џ
2ConvNet/cnn_logits_out/bias/Adam/Initializer/zerosConst*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
valueB*    *
dtype0*
_output_shapes
:
М
 ConvNet/cnn_logits_out/bias/Adam
VariableV2*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 

'ConvNet/cnn_logits_out/bias/Adam/AssignAssign ConvNet/cnn_logits_out/bias/Adam2ConvNet/cnn_logits_out/bias/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:
Ј
%ConvNet/cnn_logits_out/bias/Adam/readIdentity ConvNet/cnn_logits_out/bias/Adam*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
:
Б
4ConvNet/cnn_logits_out/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
valueB*    *
dtype0
О
"ConvNet/cnn_logits_out/bias/Adam_1
VariableV2*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 

)ConvNet/cnn_logits_out/bias/Adam_1/AssignAssign"ConvNet/cnn_logits_out/bias/Adam_14ConvNet/cnn_logits_out/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:
Ќ
'ConvNet/cnn_logits_out/bias/Adam_1/readIdentity"ConvNet/cnn_logits_out/bias/Adam_1*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0
Ё
+Adam/update_ConvNet/conv1d/kernel/ApplyAdam	ApplyAdamConvNet/conv1d/kernelConvNet/conv1d/kernel/AdamConvNet/conv1d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/ConvNet/conv1d/conv1d/ExpandDims_1_grad/Reshape*
use_nesterov( *#
_output_shapes
:*
use_locking( *
T0*(
_class
loc:@ConvNet/conv1d/kernel

)Adam/update_ConvNet/conv1d/bias/ApplyAdam	ApplyAdamConvNet/conv1d/biasConvNet/conv1d/bias/AdamConvNet/conv1d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/conv1d/BiasAdd_grad/tuple/control_dependency_1*
T0*&
_class
loc:@ConvNet/conv1d/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ў
-Adam/update_ConvNet/conv1d_1/kernel/ApplyAdam	ApplyAdamConvNet/conv1d_1/kernelConvNet/conv1d_1/kernel/AdamConvNet/conv1d_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/ConvNet/conv1d_1/conv1d/ExpandDims_1_grad/Reshape*
use_locking( *
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*
use_nesterov( *$
_output_shapes
:
Ђ
+Adam/update_ConvNet/conv1d_1/bias/ApplyAdam	ApplyAdamConvNet/conv1d_1/biasConvNet/conv1d_1/bias/AdamConvNet/conv1d_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/ConvNet/conv1d_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*(
_class
loc:@ConvNet/conv1d_1/bias
Ў
-Adam/update_ConvNet/conv1d_2/kernel/ApplyAdam	ApplyAdamConvNet/conv1d_2/kernelConvNet/conv1d_2/kernel/AdamConvNet/conv1d_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/ConvNet/conv1d_2/conv1d/ExpandDims_1_grad/Reshape*
use_locking( *
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
use_nesterov( *$
_output_shapes
:
Ђ
+Adam/update_ConvNet/conv1d_2/bias/ApplyAdam	ApplyAdamConvNet/conv1d_2/biasConvNet/conv1d_2/bias/AdamConvNet/conv1d_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/ConvNet/conv1d_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
use_nesterov( *
_output_shapes	
:
Ў
-Adam/update_ConvNet/conv1d_3/kernel/ApplyAdam	ApplyAdamConvNet/conv1d_3/kernelConvNet/conv1d_3/kernel/AdamConvNet/conv1d_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/ConvNet/conv1d_3/conv1d/ExpandDims_1_grad/Reshape*
use_nesterov( *$
_output_shapes
:*
use_locking( *
T0**
_class 
loc:@ConvNet/conv1d_3/kernel
Ђ
+Adam/update_ConvNet/conv1d_3/bias/ApplyAdam	ApplyAdamConvNet/conv1d_3/biasConvNet/conv1d_3/bias/AdamConvNet/conv1d_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/ConvNet/conv1d_3/BiasAdd_grad/tuple/control_dependency_1*(
_class
loc:@ConvNet/conv1d_3/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ў
-Adam/update_ConvNet/conv1d_4/kernel/ApplyAdam	ApplyAdamConvNet/conv1d_4/kernelConvNet/conv1d_4/kernel/AdamConvNet/conv1d_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/ConvNet/conv1d_4/conv1d/ExpandDims_1_grad/Reshape*$
_output_shapes
:*
use_locking( *
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*
use_nesterov( 
Ђ
+Adam/update_ConvNet/conv1d_4/bias/ApplyAdam	ApplyAdamConvNet/conv1d_4/biasConvNet/conv1d_4/bias/AdamConvNet/conv1d_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/ConvNet/conv1d_4/BiasAdd_grad/tuple/control_dependency_1*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ў
-Adam/update_ConvNet/conv1d_5/kernel/ApplyAdam	ApplyAdamConvNet/conv1d_5/kernelConvNet/conv1d_5/kernel/AdamConvNet/conv1d_5/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/ConvNet/conv1d_5/conv1d/ExpandDims_1_grad/Reshape*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*
use_nesterov( *$
_output_shapes
:*
use_locking( 
Ђ
+Adam/update_ConvNet/conv1d_5/bias/ApplyAdam	ApplyAdamConvNet/conv1d_5/biasConvNet/conv1d_5/bias/AdamConvNet/conv1d_5/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/ConvNet/conv1d_5/BiasAdd_grad/tuple/control_dependency_1*(
_class
loc:@ConvNet/conv1d_5/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ў
-Adam/update_ConvNet/conv1d_6/kernel/ApplyAdam	ApplyAdamConvNet/conv1d_6/kernelConvNet/conv1d_6/kernel/AdamConvNet/conv1d_6/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/ConvNet/conv1d_6/conv1d/ExpandDims_1_grad/Reshape*
use_locking( *
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
use_nesterov( *$
_output_shapes
:
Ђ
+Adam/update_ConvNet/conv1d_6/bias/ApplyAdam	ApplyAdamConvNet/conv1d_6/biasConvNet/conv1d_6/bias/AdamConvNet/conv1d_6/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/ConvNet/conv1d_6/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
use_nesterov( *
_output_shapes	
:

*Adam/update_ConvNet/dense/kernel/ApplyAdam	ApplyAdamConvNet/dense/kernelConvNet/dense/kernel/AdamConvNet/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/ConvNet/dense/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
use_locking( *
T0*'
_class
loc:@ConvNet/dense/kernel*
use_nesterov( 

(Adam/update_ConvNet/dense/bias/ApplyAdam	ApplyAdamConvNet/dense/biasConvNet/dense/bias/AdamConvNet/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependency_1*%
_class
loc:@ConvNet/dense/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Њ
,Adam/update_ConvNet/dense_1/kernel/ApplyAdam	ApplyAdamConvNet/dense_1/kernelConvNet/dense_1/kernel/AdamConvNet/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*)
_class
loc:@ConvNet/dense_1/kernel

*Adam/update_ConvNet/dense_1/bias/ApplyAdam	ApplyAdamConvNet/dense_1/biasConvNet/dense_1/bias/AdamConvNet/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@ConvNet/dense_1/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
г
3Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam	ApplyAdamConvNet/cnn_logits_out/kernel"ConvNet/cnn_logits_out/kernel/Adam$ConvNet/cnn_logits_out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
use_nesterov( *
_output_shapes
:	
Х
1Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam	ApplyAdamConvNet/cnn_logits_out/bias ConvNet/cnn_logits_out/bias/Adam"ConvNet/cnn_logits_out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
use_nesterov( *
_output_shapes
:
Њ
Adam/mulMulbeta1_power/read
Adam/beta12^Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam4^Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam*^Adam/update_ConvNet/conv1d/bias/ApplyAdam,^Adam/update_ConvNet/conv1d/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_1/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_1/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_2/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_2/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_3/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_3/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_4/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_4/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_5/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_5/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_6/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_6/kernel/ApplyAdam)^Adam/update_ConvNet/dense/bias/ApplyAdam+^Adam/update_ConvNet/dense/kernel/ApplyAdam+^Adam/update_ConvNet/dense_1/bias/ApplyAdam-^Adam/update_ConvNet/dense_1/kernel/ApplyAdam*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
: 
І
Adam/AssignAssignbeta1_powerAdam/mul*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
Ќ

Adam/mul_1Mulbeta2_power/read
Adam/beta22^Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam4^Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam*^Adam/update_ConvNet/conv1d/bias/ApplyAdam,^Adam/update_ConvNet/conv1d/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_1/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_1/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_2/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_2/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_3/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_3/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_4/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_4/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_5/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_5/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_6/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_6/kernel/ApplyAdam)^Adam/update_ConvNet/dense/bias/ApplyAdam+^Adam/update_ConvNet/dense/kernel/ApplyAdam+^Adam/update_ConvNet/dense_1/bias/ApplyAdam-^Adam/update_ConvNet/dense_1/kernel/ApplyAdam*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
: 
Њ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
ж
AdamNoOp^Adam/Assign^Adam/Assign_12^Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam4^Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam*^Adam/update_ConvNet/conv1d/bias/ApplyAdam,^Adam/update_ConvNet/conv1d/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_1/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_1/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_2/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_2/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_3/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_3/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_4/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_4/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_5/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_5/kernel/ApplyAdam,^Adam/update_ConvNet/conv1d_6/bias/ApplyAdam.^Adam/update_ConvNet/conv1d_6/kernel/ApplyAdam)^Adam/update_ConvNet/dense/bias/ApplyAdam+^Adam/update_ConvNet/dense/kernel/ApplyAdam+^Adam/update_ConvNet/dense_1/bias/ApplyAdam-^Adam/update_ConvNet/dense_1/kernel/ApplyAdam
i
soft_predictSoftmaxConvNet/cnn_logits_out/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
W
zerosConst*
valueB: *
dtype0*
_output_shapes

:
|
Variable
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 

Variable/AssignAssignVariablezeros*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(
i
Variable/readIdentityVariable*
_output_shapes

:*
T0*
_class
loc:@Variable
И
initNoOp(^ConvNet/cnn_logits_out/bias/Adam/Assign*^ConvNet/cnn_logits_out/bias/Adam_1/Assign#^ConvNet/cnn_logits_out/bias/Assign*^ConvNet/cnn_logits_out/kernel/Adam/Assign,^ConvNet/cnn_logits_out/kernel/Adam_1/Assign%^ConvNet/cnn_logits_out/kernel/Assign ^ConvNet/conv1d/bias/Adam/Assign"^ConvNet/conv1d/bias/Adam_1/Assign^ConvNet/conv1d/bias/Assign"^ConvNet/conv1d/kernel/Adam/Assign$^ConvNet/conv1d/kernel/Adam_1/Assign^ConvNet/conv1d/kernel/Assign"^ConvNet/conv1d_1/bias/Adam/Assign$^ConvNet/conv1d_1/bias/Adam_1/Assign^ConvNet/conv1d_1/bias/Assign$^ConvNet/conv1d_1/kernel/Adam/Assign&^ConvNet/conv1d_1/kernel/Adam_1/Assign^ConvNet/conv1d_1/kernel/Assign"^ConvNet/conv1d_2/bias/Adam/Assign$^ConvNet/conv1d_2/bias/Adam_1/Assign^ConvNet/conv1d_2/bias/Assign$^ConvNet/conv1d_2/kernel/Adam/Assign&^ConvNet/conv1d_2/kernel/Adam_1/Assign^ConvNet/conv1d_2/kernel/Assign"^ConvNet/conv1d_3/bias/Adam/Assign$^ConvNet/conv1d_3/bias/Adam_1/Assign^ConvNet/conv1d_3/bias/Assign$^ConvNet/conv1d_3/kernel/Adam/Assign&^ConvNet/conv1d_3/kernel/Adam_1/Assign^ConvNet/conv1d_3/kernel/Assign"^ConvNet/conv1d_4/bias/Adam/Assign$^ConvNet/conv1d_4/bias/Adam_1/Assign^ConvNet/conv1d_4/bias/Assign$^ConvNet/conv1d_4/kernel/Adam/Assign&^ConvNet/conv1d_4/kernel/Adam_1/Assign^ConvNet/conv1d_4/kernel/Assign"^ConvNet/conv1d_5/bias/Adam/Assign$^ConvNet/conv1d_5/bias/Adam_1/Assign^ConvNet/conv1d_5/bias/Assign$^ConvNet/conv1d_5/kernel/Adam/Assign&^ConvNet/conv1d_5/kernel/Adam_1/Assign^ConvNet/conv1d_5/kernel/Assign"^ConvNet/conv1d_6/bias/Adam/Assign$^ConvNet/conv1d_6/bias/Adam_1/Assign^ConvNet/conv1d_6/bias/Assign$^ConvNet/conv1d_6/kernel/Adam/Assign&^ConvNet/conv1d_6/kernel/Adam_1/Assign^ConvNet/conv1d_6/kernel/Assign^ConvNet/dense/bias/Adam/Assign!^ConvNet/dense/bias/Adam_1/Assign^ConvNet/dense/bias/Assign!^ConvNet/dense/kernel/Adam/Assign#^ConvNet/dense/kernel/Adam_1/Assign^ConvNet/dense/kernel/Assign!^ConvNet/dense_1/bias/Adam/Assign#^ConvNet/dense_1/bias/Adam_1/Assign^ConvNet/dense_1/bias/Assign#^ConvNet/dense_1/kernel/Adam/Assign%^ConvNet/dense_1/kernel/Adam_1/Assign^ConvNet/dense_1/kernel/Assign^Variable/Assign^beta1_power/Assign^beta2_power/Assign

optimizationsConst*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion*
dtype0*
_output_shapes
:
Ј

IteratorV2
IteratorV2*
output_types
2	*
shared_name *:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_output_shapes
: *
	container 

TensorSliceDatasetTensorSliceDatasetflat_filenames*
Toutput_types
2*
_class
loc:@IteratorV2*
_output_shapes
: *
output_shapes
: 
ж
FlatMapDatasetFlatMapDatasetTensorSliceDataset*
output_shapes
: *
_class
loc:@IteratorV2*)
f$R"
 Dataset_flat_map_read_one_file_4*
output_types
2*

Targuments
 *
_output_shapes
: 


MapDataset
MapDatasetFlatMapDataset*)
f$R"
 Dataset_map_transform_to_orig_10*
output_types
2	*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( * 
output_shapes
::*
_class
loc:@IteratorV2
з
ShuffleDatasetShuffleDataset
MapDatasetbuffer_sizeseedseed2*
_class
loc:@IteratorV2*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2	* 
output_shapes
::
з
BatchDatasetV2BatchDatasetV2ShuffleDataset
batch_sizedrop_remainder*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2	
Ь
PrefetchDatasetPrefetchDatasetBatchDatasetV2buffer_size_1*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2	
Э
OptimizeDatasetOptimizeDatasetPrefetchDatasetoptimizations*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2	
И
ModelDatasetModelDatasetOptimizeDataset*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2	*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ
U
MakeIteratorMakeIteratorModelDataset
IteratorV2*
_class
loc:@IteratorV2
T
IteratorToStringHandleIteratorToStringHandle
IteratorV2*
_output_shapes
: 
П
IteratorGetNextIteratorGetNext
IteratorV2*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*;
_output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
output_types
2	

optimizations_1Const*
_output_shapes
:*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion*
dtype0
Њ
IteratorV2_1
IteratorV2*
shared_name *:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_output_shapes
: *
	container *
output_types
2	
Є
TensorSliceDataset_1TensorSliceDatasetflat_filenames_1*
output_shapes
: *
Toutput_types
2*
_class
loc:@IteratorV2_1*
_output_shapes
: 
н
FlatMapDataset_1FlatMapDatasetTensorSliceDataset_1*
output_shapes
: *
_class
loc:@IteratorV2_1**
f%R#
!Dataset_flat_map_read_one_file_36*
output_types
2*

Targuments
 *
_output_shapes
: 

MapDataset_1
MapDatasetFlatMapDataset_1*
output_types
2	*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( * 
output_shapes
::*
_class
loc:@IteratorV2_1*)
f$R"
 Dataset_map_transform_to_orig_42
н
BatchDatasetV2_1BatchDatasetV2MapDataset_1batch_size_1drop_remainder_1*
output_types
2	*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2_1*
_output_shapes
: 
в
PrefetchDataset_1PrefetchDatasetBatchDatasetV2_1buffer_size_2*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2_1*
_output_shapes
: *
output_types
2	
е
OptimizeDataset_1OptimizeDatasetPrefetchDataset_1optimizations_1*
_output_shapes
: *
output_types
2	*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2_1
О
ModelDataset_1ModelDatasetOptimizeDataset_1*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
_class
loc:@IteratorV2_1*
_output_shapes
: *
output_types
2	
]
MakeIterator_1MakeIteratorModelDataset_1IteratorV2_1*
_class
loc:@IteratorV2_1
X
IteratorToStringHandle_1IteratorToStringHandleIteratorV2_1*
_output_shapes
: 
У
IteratorGetNext_1IteratorGetNextIteratorV2_1*;
_output_shapes)
':џџџџџџџџџ:џџџџџџџџџ*
output_types
2	*:
output_shapes)
':џџџџџџџџџ:џџџџџџџџџ
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_79fd857af9024de5b1883558d5d53653/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ѕ
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:?*Щ
valueПBМ?BConvNet/cnn_logits_out/biasB ConvNet/cnn_logits_out/bias/AdamB"ConvNet/cnn_logits_out/bias/Adam_1BConvNet/cnn_logits_out/kernelB"ConvNet/cnn_logits_out/kernel/AdamB$ConvNet/cnn_logits_out/kernel/Adam_1BConvNet/conv1d/biasBConvNet/conv1d/bias/AdamBConvNet/conv1d/bias/Adam_1BConvNet/conv1d/kernelBConvNet/conv1d/kernel/AdamBConvNet/conv1d/kernel/Adam_1BConvNet/conv1d_1/biasBConvNet/conv1d_1/bias/AdamBConvNet/conv1d_1/bias/Adam_1BConvNet/conv1d_1/kernelBConvNet/conv1d_1/kernel/AdamBConvNet/conv1d_1/kernel/Adam_1BConvNet/conv1d_2/biasBConvNet/conv1d_2/bias/AdamBConvNet/conv1d_2/bias/Adam_1BConvNet/conv1d_2/kernelBConvNet/conv1d_2/kernel/AdamBConvNet/conv1d_2/kernel/Adam_1BConvNet/conv1d_3/biasBConvNet/conv1d_3/bias/AdamBConvNet/conv1d_3/bias/Adam_1BConvNet/conv1d_3/kernelBConvNet/conv1d_3/kernel/AdamBConvNet/conv1d_3/kernel/Adam_1BConvNet/conv1d_4/biasBConvNet/conv1d_4/bias/AdamBConvNet/conv1d_4/bias/Adam_1BConvNet/conv1d_4/kernelBConvNet/conv1d_4/kernel/AdamBConvNet/conv1d_4/kernel/Adam_1BConvNet/conv1d_5/biasBConvNet/conv1d_5/bias/AdamBConvNet/conv1d_5/bias/Adam_1BConvNet/conv1d_5/kernelBConvNet/conv1d_5/kernel/AdamBConvNet/conv1d_5/kernel/Adam_1BConvNet/conv1d_6/biasBConvNet/conv1d_6/bias/AdamBConvNet/conv1d_6/bias/Adam_1BConvNet/conv1d_6/kernelBConvNet/conv1d_6/kernel/AdamBConvNet/conv1d_6/kernel/Adam_1BConvNet/dense/biasBConvNet/dense/bias/AdamBConvNet/dense/bias/Adam_1BConvNet/dense/kernelBConvNet/dense/kernel/AdamBConvNet/dense/kernel/Adam_1BConvNet/dense_1/biasBConvNet/dense_1/bias/AdamBConvNet/dense_1/bias/Adam_1BConvNet/dense_1/kernelBConvNet/dense_1/kernel/AdamBConvNet/dense_1/kernel/Adam_1BVariableBbeta1_powerBbeta2_power
ѓ
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
ѕ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConvNet/cnn_logits_out/bias ConvNet/cnn_logits_out/bias/Adam"ConvNet/cnn_logits_out/bias/Adam_1ConvNet/cnn_logits_out/kernel"ConvNet/cnn_logits_out/kernel/Adam$ConvNet/cnn_logits_out/kernel/Adam_1ConvNet/conv1d/biasConvNet/conv1d/bias/AdamConvNet/conv1d/bias/Adam_1ConvNet/conv1d/kernelConvNet/conv1d/kernel/AdamConvNet/conv1d/kernel/Adam_1ConvNet/conv1d_1/biasConvNet/conv1d_1/bias/AdamConvNet/conv1d_1/bias/Adam_1ConvNet/conv1d_1/kernelConvNet/conv1d_1/kernel/AdamConvNet/conv1d_1/kernel/Adam_1ConvNet/conv1d_2/biasConvNet/conv1d_2/bias/AdamConvNet/conv1d_2/bias/Adam_1ConvNet/conv1d_2/kernelConvNet/conv1d_2/kernel/AdamConvNet/conv1d_2/kernel/Adam_1ConvNet/conv1d_3/biasConvNet/conv1d_3/bias/AdamConvNet/conv1d_3/bias/Adam_1ConvNet/conv1d_3/kernelConvNet/conv1d_3/kernel/AdamConvNet/conv1d_3/kernel/Adam_1ConvNet/conv1d_4/biasConvNet/conv1d_4/bias/AdamConvNet/conv1d_4/bias/Adam_1ConvNet/conv1d_4/kernelConvNet/conv1d_4/kernel/AdamConvNet/conv1d_4/kernel/Adam_1ConvNet/conv1d_5/biasConvNet/conv1d_5/bias/AdamConvNet/conv1d_5/bias/Adam_1ConvNet/conv1d_5/kernelConvNet/conv1d_5/kernel/AdamConvNet/conv1d_5/kernel/Adam_1ConvNet/conv1d_6/biasConvNet/conv1d_6/bias/AdamConvNet/conv1d_6/bias/Adam_1ConvNet/conv1d_6/kernelConvNet/conv1d_6/kernel/AdamConvNet/conv1d_6/kernel/Adam_1ConvNet/dense/biasConvNet/dense/bias/AdamConvNet/dense/bias/Adam_1ConvNet/dense/kernelConvNet/dense/kernel/AdamConvNet/dense/kernel/Adam_1ConvNet/dense_1/biasConvNet/dense_1/bias/AdamConvNet/dense_1/bias/Adam_1ConvNet/dense_1/kernelConvNet/dense_1/kernel/AdamConvNet/dense_1/kernel/Adam_1Variablebeta1_powerbeta2_power"/device:CPU:0*M
dtypesC
A2?
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Ј
save/RestoreV2/tensor_namesConst"/device:CPU:0*Щ
valueПBМ?BConvNet/cnn_logits_out/biasB ConvNet/cnn_logits_out/bias/AdamB"ConvNet/cnn_logits_out/bias/Adam_1BConvNet/cnn_logits_out/kernelB"ConvNet/cnn_logits_out/kernel/AdamB$ConvNet/cnn_logits_out/kernel/Adam_1BConvNet/conv1d/biasBConvNet/conv1d/bias/AdamBConvNet/conv1d/bias/Adam_1BConvNet/conv1d/kernelBConvNet/conv1d/kernel/AdamBConvNet/conv1d/kernel/Adam_1BConvNet/conv1d_1/biasBConvNet/conv1d_1/bias/AdamBConvNet/conv1d_1/bias/Adam_1BConvNet/conv1d_1/kernelBConvNet/conv1d_1/kernel/AdamBConvNet/conv1d_1/kernel/Adam_1BConvNet/conv1d_2/biasBConvNet/conv1d_2/bias/AdamBConvNet/conv1d_2/bias/Adam_1BConvNet/conv1d_2/kernelBConvNet/conv1d_2/kernel/AdamBConvNet/conv1d_2/kernel/Adam_1BConvNet/conv1d_3/biasBConvNet/conv1d_3/bias/AdamBConvNet/conv1d_3/bias/Adam_1BConvNet/conv1d_3/kernelBConvNet/conv1d_3/kernel/AdamBConvNet/conv1d_3/kernel/Adam_1BConvNet/conv1d_4/biasBConvNet/conv1d_4/bias/AdamBConvNet/conv1d_4/bias/Adam_1BConvNet/conv1d_4/kernelBConvNet/conv1d_4/kernel/AdamBConvNet/conv1d_4/kernel/Adam_1BConvNet/conv1d_5/biasBConvNet/conv1d_5/bias/AdamBConvNet/conv1d_5/bias/Adam_1BConvNet/conv1d_5/kernelBConvNet/conv1d_5/kernel/AdamBConvNet/conv1d_5/kernel/Adam_1BConvNet/conv1d_6/biasBConvNet/conv1d_6/bias/AdamBConvNet/conv1d_6/bias/Adam_1BConvNet/conv1d_6/kernelBConvNet/conv1d_6/kernel/AdamBConvNet/conv1d_6/kernel/Adam_1BConvNet/dense/biasBConvNet/dense/bias/AdamBConvNet/dense/bias/Adam_1BConvNet/dense/kernelBConvNet/dense/kernel/AdamBConvNet/dense/kernel/Adam_1BConvNet/dense_1/biasBConvNet/dense_1/bias/AdamBConvNet/dense_1/bias/Adam_1BConvNet/dense_1/kernelBConvNet/dense_1/kernel/AdamBConvNet/dense_1/kernel/Adam_1BVariableBbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:?
і
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
valueB?B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
и
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapesџ
ќ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?
Р
save/AssignAssignConvNet/cnn_logits_out/biassave/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias
Щ
save/Assign_1Assign ConvNet/cnn_logits_out/bias/Adamsave/RestoreV2:1*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ы
save/Assign_2Assign"ConvNet/cnn_logits_out/bias/Adam_1save/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias
Э
save/Assign_3AssignConvNet/cnn_logits_out/kernelsave/RestoreV2:3*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
в
save/Assign_4Assign"ConvNet/cnn_logits_out/kernel/Adamsave/RestoreV2:4*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
д
save/Assign_5Assign$ConvNet/cnn_logits_out/kernel/Adam_1save/RestoreV2:5*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
Е
save/Assign_6AssignConvNet/conv1d/biassave/RestoreV2:6*
use_locking(*
T0*&
_class
loc:@ConvNet/conv1d/bias*
validate_shape(*
_output_shapes	
:
К
save/Assign_7AssignConvNet/conv1d/bias/Adamsave/RestoreV2:7*
use_locking(*
T0*&
_class
loc:@ConvNet/conv1d/bias*
validate_shape(*
_output_shapes	
:
М
save/Assign_8AssignConvNet/conv1d/bias/Adam_1save/RestoreV2:8*
use_locking(*
T0*&
_class
loc:@ConvNet/conv1d/bias*
validate_shape(*
_output_shapes	
:
С
save/Assign_9AssignConvNet/conv1d/kernelsave/RestoreV2:9*(
_class
loc:@ConvNet/conv1d/kernel*
validate_shape(*#
_output_shapes
:*
use_locking(*
T0
Ш
save/Assign_10AssignConvNet/conv1d/kernel/Adamsave/RestoreV2:10*#
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d/kernel*
validate_shape(
Ъ
save/Assign_11AssignConvNet/conv1d/kernel/Adam_1save/RestoreV2:11*
validate_shape(*#
_output_shapes
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d/kernel
Л
save/Assign_12AssignConvNet/conv1d_1/biassave/RestoreV2:12*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_1/bias*
validate_shape(*
_output_shapes	
:
Р
save/Assign_13AssignConvNet/conv1d_1/bias/Adamsave/RestoreV2:13*(
_class
loc:@ConvNet/conv1d_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Т
save/Assign_14AssignConvNet/conv1d_1/bias/Adam_1save/RestoreV2:14*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_1/bias*
validate_shape(*
_output_shapes	
:
Ш
save/Assign_15AssignConvNet/conv1d_1/kernelsave/RestoreV2:15*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*
validate_shape(
Э
save/Assign_16AssignConvNet/conv1d_1/kernel/Adamsave/RestoreV2:16*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*
validate_shape(*$
_output_shapes
:
Я
save/Assign_17AssignConvNet/conv1d_1/kernel/Adam_1save/RestoreV2:17*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_1/kernel*
validate_shape(
Л
save/Assign_18AssignConvNet/conv1d_2/biassave/RestoreV2:18*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_2/bias
Р
save/Assign_19AssignConvNet/conv1d_2/bias/Adamsave/RestoreV2:19*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
validate_shape(*
_output_shapes	
:
Т
save/Assign_20AssignConvNet/conv1d_2/bias/Adam_1save/RestoreV2:20*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_2/bias*
validate_shape(*
_output_shapes	
:
Ш
save/Assign_21AssignConvNet/conv1d_2/kernelsave/RestoreV2:21*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
validate_shape(*$
_output_shapes
:
Э
save/Assign_22AssignConvNet/conv1d_2/kernel/Adamsave/RestoreV2:22*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
validate_shape(*$
_output_shapes
:
Я
save/Assign_23AssignConvNet/conv1d_2/kernel/Adam_1save/RestoreV2:23*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_2/kernel*
validate_shape(
Л
save/Assign_24AssignConvNet/conv1d_3/biassave/RestoreV2:24*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_3/bias
Р
save/Assign_25AssignConvNet/conv1d_3/bias/Adamsave/RestoreV2:25*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
validate_shape(*
_output_shapes	
:
Т
save/Assign_26AssignConvNet/conv1d_3/bias/Adam_1save/RestoreV2:26*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_3/bias*
validate_shape(
Ш
save/Assign_27AssignConvNet/conv1d_3/kernelsave/RestoreV2:27*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
validate_shape(*$
_output_shapes
:
Э
save/Assign_28AssignConvNet/conv1d_3/kernel/Adamsave/RestoreV2:28*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
validate_shape(
Я
save/Assign_29AssignConvNet/conv1d_3/kernel/Adam_1save/RestoreV2:29*
T0**
_class 
loc:@ConvNet/conv1d_3/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(
Л
save/Assign_30AssignConvNet/conv1d_4/biassave/RestoreV2:30*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
validate_shape(*
_output_shapes	
:
Р
save/Assign_31AssignConvNet/conv1d_4/bias/Adamsave/RestoreV2:31*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
validate_shape(*
_output_shapes	
:
Т
save/Assign_32AssignConvNet/conv1d_4/bias/Adam_1save/RestoreV2:32*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_4/bias*
validate_shape(*
_output_shapes	
:
Ш
save/Assign_33AssignConvNet/conv1d_4/kernelsave/RestoreV2:33*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*
validate_shape(*$
_output_shapes
:
Э
save/Assign_34AssignConvNet/conv1d_4/kernel/Adamsave/RestoreV2:34**
_class 
loc:@ConvNet/conv1d_4/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0
Я
save/Assign_35AssignConvNet/conv1d_4/kernel/Adam_1save/RestoreV2:35*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_4/kernel*
validate_shape(
Л
save/Assign_36AssignConvNet/conv1d_5/biassave/RestoreV2:36*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_5/bias
Р
save/Assign_37AssignConvNet/conv1d_5/bias/Adamsave/RestoreV2:37*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_5/bias*
validate_shape(*
_output_shapes	
:
Т
save/Assign_38AssignConvNet/conv1d_5/bias/Adam_1save/RestoreV2:38*
T0*(
_class
loc:@ConvNet/conv1d_5/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ш
save/Assign_39AssignConvNet/conv1d_5/kernelsave/RestoreV2:39*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*
validate_shape(
Э
save/Assign_40AssignConvNet/conv1d_5/kernel/Adamsave/RestoreV2:40*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel*
validate_shape(*$
_output_shapes
:
Я
save/Assign_41AssignConvNet/conv1d_5/kernel/Adam_1save/RestoreV2:41*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_5/kernel
Л
save/Assign_42AssignConvNet/conv1d_6/biassave/RestoreV2:42*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
validate_shape(*
_output_shapes	
:
Р
save/Assign_43AssignConvNet/conv1d_6/bias/Adamsave/RestoreV2:43*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_6/bias*
validate_shape(*
_output_shapes	
:
Т
save/Assign_44AssignConvNet/conv1d_6/bias/Adam_1save/RestoreV2:44*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*(
_class
loc:@ConvNet/conv1d_6/bias
Ш
save/Assign_45AssignConvNet/conv1d_6/kernelsave/RestoreV2:45**
_class 
loc:@ConvNet/conv1d_6/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0
Э
save/Assign_46AssignConvNet/conv1d_6/kernel/Adamsave/RestoreV2:46*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel*
validate_shape(*$
_output_shapes
:*
use_locking(
Я
save/Assign_47AssignConvNet/conv1d_6/kernel/Adam_1save/RestoreV2:47*
validate_shape(*$
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@ConvNet/conv1d_6/kernel
Е
save/Assign_48AssignConvNet/dense/biassave/RestoreV2:48*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:
К
save/Assign_49AssignConvNet/dense/bias/Adamsave/RestoreV2:49*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
М
save/Assign_50AssignConvNet/dense/bias/Adam_1save/RestoreV2:50*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:
О
save/Assign_51AssignConvNet/dense/kernelsave/RestoreV2:51* 
_output_shapes
:
*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(
У
save/Assign_52AssignConvNet/dense/kernel/Adamsave/RestoreV2:52*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:

Х
save/Assign_53AssignConvNet/dense/kernel/Adam_1save/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:

Й
save/Assign_54AssignConvNet/dense_1/biassave/RestoreV2:54*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias
О
save/Assign_55AssignConvNet/dense_1/bias/Adamsave/RestoreV2:55*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:
Р
save/Assign_56AssignConvNet/dense_1/bias/Adam_1save/RestoreV2:56*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(
Т
save/Assign_57AssignConvNet/dense_1/kernelsave/RestoreV2:57*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ч
save/Assign_58AssignConvNet/dense_1/kernel/Adamsave/RestoreV2:58* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(
Щ
save/Assign_59AssignConvNet/dense_1/kernel/Adam_1save/RestoreV2:59*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Є
save/Assign_60AssignVariablesave/RestoreV2:60*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:
В
save/Assign_61Assignbeta1_powersave/RestoreV2:61*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias
В
save/Assign_62Assignbeta2_powersave/RestoreV2:62*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
Н
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shardЧ


 Dataset_flat_map_read_one_file_4
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
valueB		 R*
dtype0	Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0


!Dataset_flat_map_read_one_file_36
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
valueB		 R*
dtype0	Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0


 Dataset_map_transform_to_orig_10
arg0
reshape
	reshape_1	2DWrapper for passing nested structures to and from tf.data functions.И
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0*
Tdense
 *

num_sparse*

dense_keys
 *
sparse_types
2	*
dense_shapes
 *
sparse_keys
XYH
SparseToDense/default_valueConst*
valueB
 *    *
dtype0
SparseToDenseSparseToDense6ParseSingleExample/ParseSingleExample:sparse_indices:05ParseSingleExample/ParseSingleExample:sparse_shapes:05ParseSingleExample/ParseSingleExample:sparse_values:0$SparseToDense/default_value:output:0*
Tindices0	*
validate_indices(*
T0G
SparseToDense_1/default_valueConst*
value	B	 R *
dtype0	
SparseToDense_1SparseToDense6ParseSingleExample/ParseSingleExample:sparse_indices:15ParseSingleExample/ParseSingleExample:sparse_shapes:15ParseSingleExample/ParseSingleExample:sparse_values:1&SparseToDense_1/default_value:output:0*
Tindices0	*
validate_indices(*
T0	<
Reshape/shapeConst*
valueB:*
dtype0X
ReshapeReshapeSparseToDense:dense:0Reshape/shape:output:0*
T0*
Tshape0=
Reshape_1/shapeConst*
valueB:*
dtype0^
	Reshape_1ReshapeSparseToDense_1:dense:0Reshape_1/shape:output:0*
T0	*
Tshape0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0


 Dataset_map_transform_to_orig_42
arg0
reshape
	reshape_1	2DWrapper for passing nested structures to and from tf.data functions.И
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0*
dense_shapes
 *
sparse_types
2	*
sparse_keys
XY*
Tdense
 *

num_sparse*

dense_keys
 H
SparseToDense/default_valueConst*
valueB
 *    *
dtype0
SparseToDenseSparseToDense6ParseSingleExample/ParseSingleExample:sparse_indices:05ParseSingleExample/ParseSingleExample:sparse_shapes:05ParseSingleExample/ParseSingleExample:sparse_values:0$SparseToDense/default_value:output:0*
Tindices0	*
validate_indices(*
T0G
SparseToDense_1/default_valueConst*
value	B	 R *
dtype0	
SparseToDense_1SparseToDense6ParseSingleExample/ParseSingleExample:sparse_indices:15ParseSingleExample/ParseSingleExample:sparse_shapes:15ParseSingleExample/ParseSingleExample:sparse_values:1&SparseToDense_1/default_value:output:0*
Tindices0	*
validate_indices(*
T0	<
Reshape/shapeConst*
valueB:*
dtype0X
ReshapeReshapeSparseToDense:dense:0Reshape/shape:output:0*
Tshape0*
T0=
Reshape_1/shapeConst*
dtype0*
valueB:^
	Reshape_1ReshapeSparseToDense_1:dense:0Reshape_1/shape:output:0*
T0	*
Tshape0"
reshapeReshape:output:0"
	reshape_1Reshape_1:output:0"<
save/Const:0save/Identity:0save/restore_all (5 @F8"с
trainable_variablesЩЦ

ConvNet/conv1d/kernel:0ConvNet/conv1d/kernel/AssignConvNet/conv1d/kernel/read:022ConvNet/conv1d/kernel/Initializer/random_uniform:08
z
ConvNet/conv1d/bias:0ConvNet/conv1d/bias/AssignConvNet/conv1d/bias/read:02'ConvNet/conv1d/bias/Initializer/zeros:08

ConvNet/conv1d_1/kernel:0ConvNet/conv1d_1/kernel/AssignConvNet/conv1d_1/kernel/read:024ConvNet/conv1d_1/kernel/Initializer/random_uniform:08

ConvNet/conv1d_1/bias:0ConvNet/conv1d_1/bias/AssignConvNet/conv1d_1/bias/read:02)ConvNet/conv1d_1/bias/Initializer/zeros:08

ConvNet/conv1d_2/kernel:0ConvNet/conv1d_2/kernel/AssignConvNet/conv1d_2/kernel/read:024ConvNet/conv1d_2/kernel/Initializer/random_uniform:08

ConvNet/conv1d_2/bias:0ConvNet/conv1d_2/bias/AssignConvNet/conv1d_2/bias/read:02)ConvNet/conv1d_2/bias/Initializer/zeros:08

ConvNet/conv1d_3/kernel:0ConvNet/conv1d_3/kernel/AssignConvNet/conv1d_3/kernel/read:024ConvNet/conv1d_3/kernel/Initializer/random_uniform:08

ConvNet/conv1d_3/bias:0ConvNet/conv1d_3/bias/AssignConvNet/conv1d_3/bias/read:02)ConvNet/conv1d_3/bias/Initializer/zeros:08

ConvNet/conv1d_4/kernel:0ConvNet/conv1d_4/kernel/AssignConvNet/conv1d_4/kernel/read:024ConvNet/conv1d_4/kernel/Initializer/random_uniform:08

ConvNet/conv1d_4/bias:0ConvNet/conv1d_4/bias/AssignConvNet/conv1d_4/bias/read:02)ConvNet/conv1d_4/bias/Initializer/zeros:08

ConvNet/conv1d_5/kernel:0ConvNet/conv1d_5/kernel/AssignConvNet/conv1d_5/kernel/read:024ConvNet/conv1d_5/kernel/Initializer/random_uniform:08

ConvNet/conv1d_5/bias:0ConvNet/conv1d_5/bias/AssignConvNet/conv1d_5/bias/read:02)ConvNet/conv1d_5/bias/Initializer/zeros:08

ConvNet/conv1d_6/kernel:0ConvNet/conv1d_6/kernel/AssignConvNet/conv1d_6/kernel/read:024ConvNet/conv1d_6/kernel/Initializer/random_uniform:08

ConvNet/conv1d_6/bias:0ConvNet/conv1d_6/bias/AssignConvNet/conv1d_6/bias/read:02)ConvNet/conv1d_6/bias/Initializer/zeros:08

ConvNet/dense/kernel:0ConvNet/dense/kernel/AssignConvNet/dense/kernel/read:021ConvNet/dense/kernel/Initializer/random_uniform:08
v
ConvNet/dense/bias:0ConvNet/dense/bias/AssignConvNet/dense/bias/read:02&ConvNet/dense/bias/Initializer/zeros:08

ConvNet/dense_1/kernel:0ConvNet/dense_1/kernel/AssignConvNet/dense_1/kernel/read:023ConvNet/dense_1/kernel/Initializer/random_uniform:08
~
ConvNet/dense_1/bias:0ConvNet/dense_1/bias/AssignConvNet/dense_1/bias/read:02(ConvNet/dense_1/bias/Initializer/zeros:08
Ћ
ConvNet/cnn_logits_out/kernel:0$ConvNet/cnn_logits_out/kernel/Assign$ConvNet/cnn_logits_out/kernel/read:02:ConvNet/cnn_logits_out/kernel/Initializer/random_uniform:08

ConvNet/cnn_logits_out/bias:0"ConvNet/cnn_logits_out/bias/Assign"ConvNet/cnn_logits_out/bias/read:02/ConvNet/cnn_logits_out/bias/Initializer/zeros:08
9

Variable:0Variable/AssignVariable/read:02zeros:08"
train_op

Adam"лI
	variablesЭIЪI

ConvNet/conv1d/kernel:0ConvNet/conv1d/kernel/AssignConvNet/conv1d/kernel/read:022ConvNet/conv1d/kernel/Initializer/random_uniform:08
z
ConvNet/conv1d/bias:0ConvNet/conv1d/bias/AssignConvNet/conv1d/bias/read:02'ConvNet/conv1d/bias/Initializer/zeros:08

ConvNet/conv1d_1/kernel:0ConvNet/conv1d_1/kernel/AssignConvNet/conv1d_1/kernel/read:024ConvNet/conv1d_1/kernel/Initializer/random_uniform:08

ConvNet/conv1d_1/bias:0ConvNet/conv1d_1/bias/AssignConvNet/conv1d_1/bias/read:02)ConvNet/conv1d_1/bias/Initializer/zeros:08

ConvNet/conv1d_2/kernel:0ConvNet/conv1d_2/kernel/AssignConvNet/conv1d_2/kernel/read:024ConvNet/conv1d_2/kernel/Initializer/random_uniform:08

ConvNet/conv1d_2/bias:0ConvNet/conv1d_2/bias/AssignConvNet/conv1d_2/bias/read:02)ConvNet/conv1d_2/bias/Initializer/zeros:08

ConvNet/conv1d_3/kernel:0ConvNet/conv1d_3/kernel/AssignConvNet/conv1d_3/kernel/read:024ConvNet/conv1d_3/kernel/Initializer/random_uniform:08

ConvNet/conv1d_3/bias:0ConvNet/conv1d_3/bias/AssignConvNet/conv1d_3/bias/read:02)ConvNet/conv1d_3/bias/Initializer/zeros:08

ConvNet/conv1d_4/kernel:0ConvNet/conv1d_4/kernel/AssignConvNet/conv1d_4/kernel/read:024ConvNet/conv1d_4/kernel/Initializer/random_uniform:08

ConvNet/conv1d_4/bias:0ConvNet/conv1d_4/bias/AssignConvNet/conv1d_4/bias/read:02)ConvNet/conv1d_4/bias/Initializer/zeros:08

ConvNet/conv1d_5/kernel:0ConvNet/conv1d_5/kernel/AssignConvNet/conv1d_5/kernel/read:024ConvNet/conv1d_5/kernel/Initializer/random_uniform:08

ConvNet/conv1d_5/bias:0ConvNet/conv1d_5/bias/AssignConvNet/conv1d_5/bias/read:02)ConvNet/conv1d_5/bias/Initializer/zeros:08

ConvNet/conv1d_6/kernel:0ConvNet/conv1d_6/kernel/AssignConvNet/conv1d_6/kernel/read:024ConvNet/conv1d_6/kernel/Initializer/random_uniform:08

ConvNet/conv1d_6/bias:0ConvNet/conv1d_6/bias/AssignConvNet/conv1d_6/bias/read:02)ConvNet/conv1d_6/bias/Initializer/zeros:08

ConvNet/dense/kernel:0ConvNet/dense/kernel/AssignConvNet/dense/kernel/read:021ConvNet/dense/kernel/Initializer/random_uniform:08
v
ConvNet/dense/bias:0ConvNet/dense/bias/AssignConvNet/dense/bias/read:02&ConvNet/dense/bias/Initializer/zeros:08

ConvNet/dense_1/kernel:0ConvNet/dense_1/kernel/AssignConvNet/dense_1/kernel/read:023ConvNet/dense_1/kernel/Initializer/random_uniform:08
~
ConvNet/dense_1/bias:0ConvNet/dense_1/bias/AssignConvNet/dense_1/bias/read:02(ConvNet/dense_1/bias/Initializer/zeros:08
Ћ
ConvNet/cnn_logits_out/kernel:0$ConvNet/cnn_logits_out/kernel/Assign$ConvNet/cnn_logits_out/kernel/read:02:ConvNet/cnn_logits_out/kernel/Initializer/random_uniform:08

ConvNet/cnn_logits_out/bias:0"ConvNet/cnn_logits_out/bias/Assign"ConvNet/cnn_logits_out/bias/read:02/ConvNet/cnn_logits_out/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

ConvNet/conv1d/kernel/Adam:0!ConvNet/conv1d/kernel/Adam/Assign!ConvNet/conv1d/kernel/Adam/read:02.ConvNet/conv1d/kernel/Adam/Initializer/zeros:0

ConvNet/conv1d/kernel/Adam_1:0#ConvNet/conv1d/kernel/Adam_1/Assign#ConvNet/conv1d/kernel/Adam_1/read:020ConvNet/conv1d/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d/bias/Adam:0ConvNet/conv1d/bias/Adam/AssignConvNet/conv1d/bias/Adam/read:02,ConvNet/conv1d/bias/Adam/Initializer/zeros:0

ConvNet/conv1d/bias/Adam_1:0!ConvNet/conv1d/bias/Adam_1/Assign!ConvNet/conv1d/bias/Adam_1/read:02.ConvNet/conv1d/bias/Adam_1/Initializer/zeros:0

ConvNet/conv1d_1/kernel/Adam:0#ConvNet/conv1d_1/kernel/Adam/Assign#ConvNet/conv1d_1/kernel/Adam/read:020ConvNet/conv1d_1/kernel/Adam/Initializer/zeros:0
Є
 ConvNet/conv1d_1/kernel/Adam_1:0%ConvNet/conv1d_1/kernel/Adam_1/Assign%ConvNet/conv1d_1/kernel/Adam_1/read:022ConvNet/conv1d_1/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d_1/bias/Adam:0!ConvNet/conv1d_1/bias/Adam/Assign!ConvNet/conv1d_1/bias/Adam/read:02.ConvNet/conv1d_1/bias/Adam/Initializer/zeros:0

ConvNet/conv1d_1/bias/Adam_1:0#ConvNet/conv1d_1/bias/Adam_1/Assign#ConvNet/conv1d_1/bias/Adam_1/read:020ConvNet/conv1d_1/bias/Adam_1/Initializer/zeros:0

ConvNet/conv1d_2/kernel/Adam:0#ConvNet/conv1d_2/kernel/Adam/Assign#ConvNet/conv1d_2/kernel/Adam/read:020ConvNet/conv1d_2/kernel/Adam/Initializer/zeros:0
Є
 ConvNet/conv1d_2/kernel/Adam_1:0%ConvNet/conv1d_2/kernel/Adam_1/Assign%ConvNet/conv1d_2/kernel/Adam_1/read:022ConvNet/conv1d_2/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d_2/bias/Adam:0!ConvNet/conv1d_2/bias/Adam/Assign!ConvNet/conv1d_2/bias/Adam/read:02.ConvNet/conv1d_2/bias/Adam/Initializer/zeros:0

ConvNet/conv1d_2/bias/Adam_1:0#ConvNet/conv1d_2/bias/Adam_1/Assign#ConvNet/conv1d_2/bias/Adam_1/read:020ConvNet/conv1d_2/bias/Adam_1/Initializer/zeros:0

ConvNet/conv1d_3/kernel/Adam:0#ConvNet/conv1d_3/kernel/Adam/Assign#ConvNet/conv1d_3/kernel/Adam/read:020ConvNet/conv1d_3/kernel/Adam/Initializer/zeros:0
Є
 ConvNet/conv1d_3/kernel/Adam_1:0%ConvNet/conv1d_3/kernel/Adam_1/Assign%ConvNet/conv1d_3/kernel/Adam_1/read:022ConvNet/conv1d_3/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d_3/bias/Adam:0!ConvNet/conv1d_3/bias/Adam/Assign!ConvNet/conv1d_3/bias/Adam/read:02.ConvNet/conv1d_3/bias/Adam/Initializer/zeros:0

ConvNet/conv1d_3/bias/Adam_1:0#ConvNet/conv1d_3/bias/Adam_1/Assign#ConvNet/conv1d_3/bias/Adam_1/read:020ConvNet/conv1d_3/bias/Adam_1/Initializer/zeros:0

ConvNet/conv1d_4/kernel/Adam:0#ConvNet/conv1d_4/kernel/Adam/Assign#ConvNet/conv1d_4/kernel/Adam/read:020ConvNet/conv1d_4/kernel/Adam/Initializer/zeros:0
Є
 ConvNet/conv1d_4/kernel/Adam_1:0%ConvNet/conv1d_4/kernel/Adam_1/Assign%ConvNet/conv1d_4/kernel/Adam_1/read:022ConvNet/conv1d_4/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d_4/bias/Adam:0!ConvNet/conv1d_4/bias/Adam/Assign!ConvNet/conv1d_4/bias/Adam/read:02.ConvNet/conv1d_4/bias/Adam/Initializer/zeros:0

ConvNet/conv1d_4/bias/Adam_1:0#ConvNet/conv1d_4/bias/Adam_1/Assign#ConvNet/conv1d_4/bias/Adam_1/read:020ConvNet/conv1d_4/bias/Adam_1/Initializer/zeros:0

ConvNet/conv1d_5/kernel/Adam:0#ConvNet/conv1d_5/kernel/Adam/Assign#ConvNet/conv1d_5/kernel/Adam/read:020ConvNet/conv1d_5/kernel/Adam/Initializer/zeros:0
Є
 ConvNet/conv1d_5/kernel/Adam_1:0%ConvNet/conv1d_5/kernel/Adam_1/Assign%ConvNet/conv1d_5/kernel/Adam_1/read:022ConvNet/conv1d_5/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d_5/bias/Adam:0!ConvNet/conv1d_5/bias/Adam/Assign!ConvNet/conv1d_5/bias/Adam/read:02.ConvNet/conv1d_5/bias/Adam/Initializer/zeros:0

ConvNet/conv1d_5/bias/Adam_1:0#ConvNet/conv1d_5/bias/Adam_1/Assign#ConvNet/conv1d_5/bias/Adam_1/read:020ConvNet/conv1d_5/bias/Adam_1/Initializer/zeros:0

ConvNet/conv1d_6/kernel/Adam:0#ConvNet/conv1d_6/kernel/Adam/Assign#ConvNet/conv1d_6/kernel/Adam/read:020ConvNet/conv1d_6/kernel/Adam/Initializer/zeros:0
Є
 ConvNet/conv1d_6/kernel/Adam_1:0%ConvNet/conv1d_6/kernel/Adam_1/Assign%ConvNet/conv1d_6/kernel/Adam_1/read:022ConvNet/conv1d_6/kernel/Adam_1/Initializer/zeros:0

ConvNet/conv1d_6/bias/Adam:0!ConvNet/conv1d_6/bias/Adam/Assign!ConvNet/conv1d_6/bias/Adam/read:02.ConvNet/conv1d_6/bias/Adam/Initializer/zeros:0

ConvNet/conv1d_6/bias/Adam_1:0#ConvNet/conv1d_6/bias/Adam_1/Assign#ConvNet/conv1d_6/bias/Adam_1/read:020ConvNet/conv1d_6/bias/Adam_1/Initializer/zeros:0

ConvNet/dense/kernel/Adam:0 ConvNet/dense/kernel/Adam/Assign ConvNet/dense/kernel/Adam/read:02-ConvNet/dense/kernel/Adam/Initializer/zeros:0

ConvNet/dense/kernel/Adam_1:0"ConvNet/dense/kernel/Adam_1/Assign"ConvNet/dense/kernel/Adam_1/read:02/ConvNet/dense/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense/bias/Adam:0ConvNet/dense/bias/Adam/AssignConvNet/dense/bias/Adam/read:02+ConvNet/dense/bias/Adam/Initializer/zeros:0

ConvNet/dense/bias/Adam_1:0 ConvNet/dense/bias/Adam_1/Assign ConvNet/dense/bias/Adam_1/read:02-ConvNet/dense/bias/Adam_1/Initializer/zeros:0

ConvNet/dense_1/kernel/Adam:0"ConvNet/dense_1/kernel/Adam/Assign"ConvNet/dense_1/kernel/Adam/read:02/ConvNet/dense_1/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_1/kernel/Adam_1:0$ConvNet/dense_1/kernel/Adam_1/Assign$ConvNet/dense_1/kernel/Adam_1/read:021ConvNet/dense_1/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_1/bias/Adam:0 ConvNet/dense_1/bias/Adam/Assign ConvNet/dense_1/bias/Adam/read:02-ConvNet/dense_1/bias/Adam/Initializer/zeros:0

ConvNet/dense_1/bias/Adam_1:0"ConvNet/dense_1/bias/Adam_1/Assign"ConvNet/dense_1/bias/Adam_1/read:02/ConvNet/dense_1/bias/Adam_1/Initializer/zeros:0
Д
$ConvNet/cnn_logits_out/kernel/Adam:0)ConvNet/cnn_logits_out/kernel/Adam/Assign)ConvNet/cnn_logits_out/kernel/Adam/read:026ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros:0
М
&ConvNet/cnn_logits_out/kernel/Adam_1:0+ConvNet/cnn_logits_out/kernel/Adam_1/Assign+ConvNet/cnn_logits_out/kernel/Adam_1/read:028ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros:0
Ќ
"ConvNet/cnn_logits_out/bias/Adam:0'ConvNet/cnn_logits_out/bias/Adam/Assign'ConvNet/cnn_logits_out/bias/Adam/read:024ConvNet/cnn_logits_out/bias/Adam/Initializer/zeros:0
Д
$ConvNet/cnn_logits_out/bias/Adam_1:0)ConvNet/cnn_logits_out/bias/Adam_1/Assign)ConvNet/cnn_logits_out/bias/Adam_1/read:026ConvNet/cnn_logits_out/bias/Adam_1/Initializer/zeros:0
9

Variable:0Variable/AssignVariable/read:02zeros:08"-
	iterators 

IteratorV2:0
IteratorV2_1:0