Îí	
ŕ,ľ,
:
Add
x"T
y"T
z"T"
Ttype:
2	
î
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
ˇ
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
ű

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
H
ShardedFilename
basename	
shard

num_shards
filename
˝
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
2
StopGradient

input"T
output"T"	
Ttype
ö
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
Ttype"fuck*1.13.12b'v1.13.1-0-g6612da8951'Ţé
Ś
ConstConst*đ
valuećBă BÜ../../data_exploration/datasets/32PSK_16APSK_32QAM_FM_GMSK_32APSK_OQPSK_8ASK_BPSK_8PSK_AM-SSB-SC_4ASK_16PSK_64APSK_128QAM_128APSK_AM-DSB-SC_AM-SSB-WC_64QAM_QPSK_256QAM_AM-DSB-WC_OOK_16QAM-20_-10_0_10_20_30_train.tfrecord*
dtype0*
_output_shapes
: 
g
flat_filenames/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
i
flat_filenamesReshapeConstflat_filenames/shape*
T0*
Tshape0*
_output_shapes
:
N
buffer_sizeConst*
value
B	 R'*
dtype0	*
_output_shapes
: 
F
seedConst*
value	B	 R *
dtype0	*
_output_shapes
: 
G
seed2Const*
dtype0	*
_output_shapes
: *
value	B	 R 
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
buffer_size_1Const*
_output_shapes
: *
value	B	 Rd*
dtype0	
§
Const_1Const*ď
valueĺBâ BŰ../../data_exploration/datasets/32PSK_16APSK_32QAM_FM_GMSK_32APSK_OQPSK_8ASK_BPSK_8PSK_AM-SSB-SC_4ASK_16PSK_64APSK_128QAM_128APSK_AM-DSB-SC_AM-SSB-WC_64QAM_QPSK_256QAM_AM-DSB-WC_OOK_16QAM-20_-10_0_10_20_30_test.tfrecord*
dtype0*
_output_shapes
: 
i
flat_filenames_1/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
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
batch_size_1Const*
value	B	 Rd*
dtype0	*
_output_shapes
: 
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
x_placeholderPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
p
y_placeholderPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
j
ConvNet/Reshape/shapeConst*!
valueB"˙˙˙˙      *
dtype0*
_output_shapes
:

ConvNet/ReshapeReshapex_placeholderConvNet/Reshape/shape*
T0*
Tshape0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
ConvNet/Flatten/flatten/ShapeShapeConvNet/Reshape*
T0*
out_type0*
_output_shapes
:
u
+ConvNet/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
-ConvNet/Flatten/flatten/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
w
-ConvNet/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ń
%ConvNet/Flatten/flatten/strided_sliceStridedSliceConvNet/Flatten/flatten/Shape+ConvNet/Flatten/flatten/strided_slice/stack-ConvNet/Flatten/flatten/strided_slice/stack_1-ConvNet/Flatten/flatten/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
r
'ConvNet/Flatten/flatten/Reshape/shape/1Const*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ˇ
%ConvNet/Flatten/flatten/Reshape/shapePack%ConvNet/Flatten/flatten/strided_slice'ConvNet/Flatten/flatten/Reshape/shape/1*
N*
_output_shapes
:*
T0*

axis 
Ł
ConvNet/Flatten/flatten/ReshapeReshapeConvNet/Reshape%ConvNet/Flatten/flatten/Reshape/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ż
5ConvNet/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *'
_class
loc:@ConvNet/dense/kernel*
dtype0*
_output_shapes
:
Ą
3ConvNet/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ěQ˝*'
_class
loc:@ConvNet/dense/kernel*
dtype0*
_output_shapes
: 
Ą
3ConvNet/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *ěQ=*'
_class
loc:@ConvNet/dense/kernel*
dtype0*
_output_shapes
: 
˙
=ConvNet/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5ConvNet/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*'
_class
loc:@ConvNet/dense/kernel*
seed2 
î
3ConvNet/dense/kernel/Initializer/random_uniform/subSub3ConvNet/dense/kernel/Initializer/random_uniform/max3ConvNet/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@ConvNet/dense/kernel*
_output_shapes
: 

3ConvNet/dense/kernel/Initializer/random_uniform/mulMul=ConvNet/dense/kernel/Initializer/random_uniform/RandomUniform3ConvNet/dense/kernel/Initializer/random_uniform/sub*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:
*
T0
ô
/ConvNet/dense/kernel/Initializer/random_uniformAdd3ConvNet/dense/kernel/Initializer/random_uniform/mul3ConvNet/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:

ľ
ConvNet/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *'
_class
loc:@ConvNet/dense/kernel*
	container *
shape:

é
ConvNet/dense/kernel/AssignAssignConvNet/dense/kernel/ConvNet/dense/kernel/Initializer/random_uniform*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

ConvNet/dense/kernel/readIdentityConvNet/dense/kernel* 
_output_shapes
:
*
T0*'
_class
loc:@ConvNet/dense/kernel

$ConvNet/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *%
_class
loc:@ConvNet/dense/bias
§
ConvNet/dense/bias
VariableV2*
shared_name *%
_class
loc:@ConvNet/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ó
ConvNet/dense/bias/AssignAssignConvNet/dense/bias$ConvNet/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias

ConvNet/dense/bias/readIdentityConvNet/dense/bias*
T0*%
_class
loc:@ConvNet/dense/bias*
_output_shapes	
:
ł
ConvNet/dense/MatMulMatMulConvNet/Flatten/flatten/ReshapeConvNet/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

ConvNet/dense/BiasAddBiasAddConvNet/dense/MatMulConvNet/dense/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
d
ConvNet/dense/ReluReluConvNet/dense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ł
7ConvNet/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *)
_class
loc:@ConvNet/dense_1/kernel
Ľ
5ConvNet/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_1/kernel*
dtype0
Ľ
5ConvNet/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_1/kernel

?ConvNet/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_1/kernel
ö
5ConvNet/dense_1/kernel/Initializer/random_uniform/subSub5ConvNet/dense_1/kernel/Initializer/random_uniform/max5ConvNet/dense_1/kernel/Initializer/random_uniform/min*)
_class
loc:@ConvNet/dense_1/kernel*
_output_shapes
: *
T0

5ConvNet/dense_1/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_1/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:

ü
1ConvNet/dense_1/kernel/Initializer/random_uniformAdd5ConvNet/dense_1/kernel/Initializer/random_uniform/mul5ConvNet/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_1/kernel
š
ConvNet/dense_1/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_1/kernel*
	container 
ń
ConvNet/dense_1/kernel/AssignAssignConvNet/dense_1/kernel1ConvNet/dense_1/kernel/Initializer/random_uniform*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

ConvNet/dense_1/kernel/readIdentityConvNet/dense_1/kernel*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:


&ConvNet/dense_1/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@ConvNet/dense_1/bias*
dtype0*
_output_shapes	
:
Ť
ConvNet/dense_1/bias
VariableV2*'
_class
loc:@ConvNet/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ű
ConvNet/dense_1/bias/AssignAssignConvNet/dense_1/bias&ConvNet/dense_1/bias/Initializer/zeros*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

ConvNet/dense_1/bias/readIdentityConvNet/dense_1/bias*
T0*'
_class
loc:@ConvNet/dense_1/bias*
_output_shapes	
:
Ş
ConvNet/dense_1/MatMulMatMulConvNet/dense/ReluConvNet/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

ConvNet/dense_1/BiasAddBiasAddConvNet/dense_1/MatMulConvNet/dense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ConvNet/dense_1/ReluReluConvNet/dense_1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7ConvNet/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_2/kernel*
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_2/kernel*
dtype0*
_output_shapes
: 
Ľ
5ConvNet/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_2/kernel*
dtype0*
_output_shapes
: 

?ConvNet/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_2/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_2/kernel*
seed2 
ö
5ConvNet/dense_2/kernel/Initializer/random_uniform/subSub5ConvNet/dense_2/kernel/Initializer/random_uniform/max5ConvNet/dense_2/kernel/Initializer/random_uniform/min*)
_class
loc:@ConvNet/dense_2/kernel*
_output_shapes
: *
T0

5ConvNet/dense_2/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_2/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@ConvNet/dense_2/kernel* 
_output_shapes
:

ü
1ConvNet/dense_2/kernel/Initializer/random_uniformAdd5ConvNet/dense_2/kernel/Initializer/random_uniform/mul5ConvNet/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_2/kernel* 
_output_shapes
:

š
ConvNet/dense_2/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_2/kernel*
	container *
shape:

ń
ConvNet/dense_2/kernel/AssignAssignConvNet/dense_2/kernel1ConvNet/dense_2/kernel/Initializer/random_uniform*
T0*)
_class
loc:@ConvNet/dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

ConvNet/dense_2/kernel/readIdentityConvNet/dense_2/kernel* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_2/kernel

&ConvNet/dense_2/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *'
_class
loc:@ConvNet/dense_2/bias*
dtype0
Ť
ConvNet/dense_2/bias
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
ConvNet/dense_2/bias/AssignAssignConvNet/dense_2/bias&ConvNet/dense_2/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_2/bias*
validate_shape(

ConvNet/dense_2/bias/readIdentityConvNet/dense_2/bias*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_2/bias
Ź
ConvNet/dense_2/MatMulMatMulConvNet/dense_1/ReluConvNet/dense_2/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

ConvNet/dense_2/BiasAddBiasAddConvNet/dense_2/MatMulConvNet/dense_2/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
ConvNet/dense_2/ReluReluConvNet/dense_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7ConvNet/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_3/kernel*
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_3/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_3/kernel*
dtype0*
_output_shapes
: 
Ľ
5ConvNet/dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_3/kernel*
dtype0*
_output_shapes
: 

?ConvNet/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_3/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_3/kernel*
seed2 
ö
5ConvNet/dense_3/kernel/Initializer/random_uniform/subSub5ConvNet/dense_3/kernel/Initializer/random_uniform/max5ConvNet/dense_3/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_3/kernel*
_output_shapes
: 

5ConvNet/dense_3/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_3/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_3/kernel/Initializer/random_uniform/sub*)
_class
loc:@ConvNet/dense_3/kernel* 
_output_shapes
:
*
T0
ü
1ConvNet/dense_3/kernel/Initializer/random_uniformAdd5ConvNet/dense_3/kernel/Initializer/random_uniform/mul5ConvNet/dense_3/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_3/kernel* 
_output_shapes
:

š
ConvNet/dense_3/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_3/kernel*
	container *
shape:

ń
ConvNet/dense_3/kernel/AssignAssignConvNet/dense_3/kernel1ConvNet/dense_3/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_3/kernel

ConvNet/dense_3/kernel/readIdentityConvNet/dense_3/kernel* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_3/kernel

&ConvNet/dense_3/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@ConvNet/dense_3/bias*
dtype0*
_output_shapes	
:
Ť
ConvNet/dense_3/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_3/bias*
	container 
Ű
ConvNet/dense_3/bias/AssignAssignConvNet/dense_3/bias&ConvNet/dense_3/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_3/bias*
validate_shape(

ConvNet/dense_3/bias/readIdentityConvNet/dense_3/bias*
T0*'
_class
loc:@ConvNet/dense_3/bias*
_output_shapes	
:
Ź
ConvNet/dense_3/MatMulMatMulConvNet/dense_2/ReluConvNet/dense_3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

ConvNet/dense_3/BiasAddBiasAddConvNet/dense_3/MatMulConvNet/dense_3/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
h
ConvNet/dense_3/ReluReluConvNet/dense_3/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ł
7ConvNet/dense_4/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_4/kernel*
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_4/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_4/kernel*
dtype0*
_output_shapes
: 
Ľ
5ConvNet/dense_4/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_4/kernel

?ConvNet/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_4/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_4/kernel*
seed2 
ö
5ConvNet/dense_4/kernel/Initializer/random_uniform/subSub5ConvNet/dense_4/kernel/Initializer/random_uniform/max5ConvNet/dense_4/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_4/kernel*
_output_shapes
: 

5ConvNet/dense_4/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_4/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_4/kernel/Initializer/random_uniform/sub*)
_class
loc:@ConvNet/dense_4/kernel* 
_output_shapes
:
*
T0
ü
1ConvNet/dense_4/kernel/Initializer/random_uniformAdd5ConvNet/dense_4/kernel/Initializer/random_uniform/mul5ConvNet/dense_4/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_4/kernel
š
ConvNet/dense_4/kernel
VariableV2* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_4/kernel*
	container *
shape:
*
dtype0
ń
ConvNet/dense_4/kernel/AssignAssignConvNet/dense_4/kernel1ConvNet/dense_4/kernel/Initializer/random_uniform*)
_class
loc:@ConvNet/dense_4/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

ConvNet/dense_4/kernel/readIdentityConvNet/dense_4/kernel*
T0*)
_class
loc:@ConvNet/dense_4/kernel* 
_output_shapes
:


&ConvNet/dense_4/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@ConvNet/dense_4/bias*
dtype0*
_output_shapes	
:
Ť
ConvNet/dense_4/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_4/bias*
	container 
Ű
ConvNet/dense_4/bias/AssignAssignConvNet/dense_4/bias&ConvNet/dense_4/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_4/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_4/bias/readIdentityConvNet/dense_4/bias*
T0*'
_class
loc:@ConvNet/dense_4/bias*
_output_shapes	
:
Ź
ConvNet/dense_4/MatMulMatMulConvNet/dense_3/ReluConvNet/dense_4/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

ConvNet/dense_4/BiasAddBiasAddConvNet/dense_4/MatMulConvNet/dense_4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ConvNet/dense_4/ReluReluConvNet/dense_4/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ł
7ConvNet/dense_5/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_5/kernel*
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_5/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_5/kernel*
dtype0*
_output_shapes
: 
Ľ
5ConvNet/dense_5/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_5/kernel*
dtype0*
_output_shapes
: 

?ConvNet/dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_5/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_5/kernel*
seed2 
ö
5ConvNet/dense_5/kernel/Initializer/random_uniform/subSub5ConvNet/dense_5/kernel/Initializer/random_uniform/max5ConvNet/dense_5/kernel/Initializer/random_uniform/min*)
_class
loc:@ConvNet/dense_5/kernel*
_output_shapes
: *
T0

5ConvNet/dense_5/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_5/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_5/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_5/kernel
ü
1ConvNet/dense_5/kernel/Initializer/random_uniformAdd5ConvNet/dense_5/kernel/Initializer/random_uniform/mul5ConvNet/dense_5/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_5/kernel
š
ConvNet/dense_5/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_5/kernel*
	container 
ń
ConvNet/dense_5/kernel/AssignAssignConvNet/dense_5/kernel1ConvNet/dense_5/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_5/kernel

ConvNet/dense_5/kernel/readIdentityConvNet/dense_5/kernel*
T0*)
_class
loc:@ConvNet/dense_5/kernel* 
_output_shapes
:


&ConvNet/dense_5/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@ConvNet/dense_5/bias*
dtype0*
_output_shapes	
:
Ť
ConvNet/dense_5/bias
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_5/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
ConvNet/dense_5/bias/AssignAssignConvNet/dense_5/bias&ConvNet/dense_5/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_5/bias

ConvNet/dense_5/bias/readIdentityConvNet/dense_5/bias*
T0*'
_class
loc:@ConvNet/dense_5/bias*
_output_shapes	
:
Ź
ConvNet/dense_5/MatMulMatMulConvNet/dense_4/ReluConvNet/dense_5/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

ConvNet/dense_5/BiasAddBiasAddConvNet/dense_5/MatMulConvNet/dense_5/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
h
ConvNet/dense_5/ReluReluConvNet/dense_5/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7ConvNet/dense_6/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_6/kernel*
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_6/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_6/kernel*
dtype0*
_output_shapes
: 
Ľ
5ConvNet/dense_6/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_6/kernel*
dtype0*
_output_shapes
: 

?ConvNet/dense_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_6/kernel/Initializer/random_uniform/shape*)
_class
loc:@ConvNet/dense_6/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
ö
5ConvNet/dense_6/kernel/Initializer/random_uniform/subSub5ConvNet/dense_6/kernel/Initializer/random_uniform/max5ConvNet/dense_6/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_6/kernel*
_output_shapes
: 

5ConvNet/dense_6/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_6/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_6/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_6/kernel
ü
1ConvNet/dense_6/kernel/Initializer/random_uniformAdd5ConvNet/dense_6/kernel/Initializer/random_uniform/mul5ConvNet/dense_6/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@ConvNet/dense_6/kernel* 
_output_shapes
:

š
ConvNet/dense_6/kernel
VariableV2*)
_class
loc:@ConvNet/dense_6/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ń
ConvNet/dense_6/kernel/AssignAssignConvNet/dense_6/kernel1ConvNet/dense_6/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_6/kernel*
validate_shape(* 
_output_shapes
:


ConvNet/dense_6/kernel/readIdentityConvNet/dense_6/kernel*
T0*)
_class
loc:@ConvNet/dense_6/kernel* 
_output_shapes
:


&ConvNet/dense_6/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *'
_class
loc:@ConvNet/dense_6/bias
Ť
ConvNet/dense_6/bias
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_6/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
ConvNet/dense_6/bias/AssignAssignConvNet/dense_6/bias&ConvNet/dense_6/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_6/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_6/bias/readIdentityConvNet/dense_6/bias*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_6/bias
Ź
ConvNet/dense_6/MatMulMatMulConvNet/dense_5/ReluConvNet/dense_6/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

ConvNet/dense_6/BiasAddBiasAddConvNet/dense_6/MatMulConvNet/dense_6/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ConvNet/dense_6/ReluReluConvNet/dense_6/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7ConvNet/dense_7/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@ConvNet/dense_7/kernel*
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_7/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*)
_class
loc:@ConvNet/dense_7/kernel*
dtype0*
_output_shapes
: 
Ľ
5ConvNet/dense_7/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*)
_class
loc:@ConvNet/dense_7/kernel*
dtype0*
_output_shapes
: 

?ConvNet/dense_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform7ConvNet/dense_7/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@ConvNet/dense_7/kernel*
seed2 
ö
5ConvNet/dense_7/kernel/Initializer/random_uniform/subSub5ConvNet/dense_7/kernel/Initializer/random_uniform/max5ConvNet/dense_7/kernel/Initializer/random_uniform/min*)
_class
loc:@ConvNet/dense_7/kernel*
_output_shapes
: *
T0

5ConvNet/dense_7/kernel/Initializer/random_uniform/mulMul?ConvNet/dense_7/kernel/Initializer/random_uniform/RandomUniform5ConvNet/dense_7/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@ConvNet/dense_7/kernel* 
_output_shapes
:

ü
1ConvNet/dense_7/kernel/Initializer/random_uniformAdd5ConvNet/dense_7/kernel/Initializer/random_uniform/mul5ConvNet/dense_7/kernel/Initializer/random_uniform/min*)
_class
loc:@ConvNet/dense_7/kernel* 
_output_shapes
:
*
T0
š
ConvNet/dense_7/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_7/kernel*
	container *
shape:

ń
ConvNet/dense_7/kernel/AssignAssignConvNet/dense_7/kernel1ConvNet/dense_7/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_7/kernel*
validate_shape(* 
_output_shapes
:


ConvNet/dense_7/kernel/readIdentityConvNet/dense_7/kernel*
T0*)
_class
loc:@ConvNet/dense_7/kernel* 
_output_shapes
:


&ConvNet/dense_7/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@ConvNet/dense_7/bias*
dtype0*
_output_shapes	
:
Ť
ConvNet/dense_7/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_7/bias*
	container *
shape:
Ű
ConvNet/dense_7/bias/AssignAssignConvNet/dense_7/bias&ConvNet/dense_7/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_7/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_7/bias/readIdentityConvNet/dense_7/bias*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_7/bias
Ź
ConvNet/dense_7/MatMulMatMulConvNet/dense_6/ReluConvNet/dense_7/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

ConvNet/dense_7/BiasAddBiasAddConvNet/dense_7/MatMulConvNet/dense_7/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
ConvNet/dense_7/ReluReluConvNet/dense_7/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
>ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/shapeConst*
valueB"      *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
dtype0*
_output_shapes
:
ł
<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ôĺž*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel
ł
<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/maxConst*
valueB
 *ôĺ>*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
dtype0*
_output_shapes
: 

FConvNet/cnn_logits_out/kernel/Initializer/random_uniform/RandomUniformRandomUniform>ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/shape*
_output_shapes
:	*

seed *
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
seed2 *
dtype0

<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/subSub<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/max<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
: 
Ľ
<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/mulMulFConvNet/cnn_logits_out/kernel/Initializer/random_uniform/RandomUniform<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/sub*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	*
T0

8ConvNet/cnn_logits_out/kernel/Initializer/random_uniformAdd<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/mul<ConvNet/cnn_logits_out/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	
Ĺ
ConvNet/cnn_logits_out/kernel
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
	container 

$ConvNet/cnn_logits_out/kernel/AssignAssignConvNet/cnn_logits_out/kernel8ConvNet/cnn_logits_out/kernel/Initializer/random_uniform*
_output_shapes
:	*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(
Š
"ConvNet/cnn_logits_out/kernel/readIdentityConvNet/cnn_logits_out/kernel*
_output_shapes
:	*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel
Ş
-ConvNet/cnn_logits_out/bias/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
dtype0*
_output_shapes
:
ˇ
ConvNet/cnn_logits_out/bias
VariableV2*
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ö
"ConvNet/cnn_logits_out/bias/AssignAssignConvNet/cnn_logits_out/bias-ConvNet/cnn_logits_out/bias/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:

 ConvNet/cnn_logits_out/bias/readIdentityConvNet/cnn_logits_out/bias*
_output_shapes
:*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias
š
ConvNet/cnn_logits_out/MatMulMatMulConvNet/dense_7/Relu"ConvNet/cnn_logits_out/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ł
ConvNet/cnn_logits_out/BiasAddBiasAddConvNet/cnn_logits_out/MatMul ConvNet/cnn_logits_out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradienty_placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
Š
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
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
ö
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
Index0*
T0

4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
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
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Í
,softmax_cross_entropy_with_logits_sg/ReshapeReshapeConvNet/cnn_logits_out/BiasAdd+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
m
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ľ
,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
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
ü
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
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
ě
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
í
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
n
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ť
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
N*
_output_shapes
:*
T0*

axis 
ú
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
É
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0

gradients/Mean_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Mean_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
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
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
out_type0*
_output_shapes
:*
T0
î
Egradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivCgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Bgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeBgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0
á
7gradients/softmax_cross_entropy_with_logits_sg_grad/mulMul>gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
ľ
>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
š
7gradients/softmax_cross_entropy_with_logits_sg_grad/NegNeg>gradients/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0

Dgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsEgradients/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0
ö
9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1Mul@gradients/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_17gradients/softmax_cross_entropy_with_logits_sg_grad/Neg*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
Â
Dgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp8^gradients/softmax_cross_entropy_with_logits_sg_grad/mul:^gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1
ß
Lgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity7gradients/softmax_cross_entropy_with_logits_sg_grad/mulE^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ĺ
Ngradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity9gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1E^gradients/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Agradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeConvNet/cnn_logits_out/BiasAdd*
_output_shapes
:*
T0*
out_type0

Cgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeLgradients/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyAgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
9gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0
Č
>gradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/group_depsNoOp:^gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGradD^gradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
â
Fgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape?^gradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
Hgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGrad?^gradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/ConvNet/cnn_logits_out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

3gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMulMatMulFgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency"ConvNet/cnn_logits_out/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
í
5gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1MatMulConvNet/dense_7/ReluFgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
ł
=gradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/group_depsNoOp4^gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul6^gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1
Á
Egradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependencyIdentity3gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul>^gradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Ggradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependency_1Identity5gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1>^gradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/ConvNet/cnn_logits_out/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Č
,gradients/ConvNet/dense_7/Relu_grad/ReluGradReluGradEgradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependencyConvNet/dense_7/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/ConvNet/dense_7/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_7/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/ConvNet/dense_7/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_7/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_7/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_7/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_7/Relu_grad/ReluGrad8^gradients/ConvNet/dense_7/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_7/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients/ConvNet/dense_7/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_7/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_7/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ConvNet/dense_7/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
,gradients/ConvNet/dense_7/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_7/BiasAdd_grad/tuple/control_dependencyConvNet/dense_7/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ŕ
.gradients/ConvNet/dense_7/MatMul_grad/MatMul_1MatMulConvNet/dense_6/Relu?gradients/ConvNet/dense_7/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

6gradients/ConvNet/dense_7/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_7/MatMul_grad/MatMul/^gradients/ConvNet/dense_7/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_7/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_7/MatMul_grad/MatMul7^gradients/ConvNet/dense_7/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_7/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/ConvNet/dense_7/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_7/MatMul_grad/MatMul_17^gradients/ConvNet/dense_7/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/ConvNet/dense_7/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/ConvNet/dense_6/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_7/MatMul_grad/tuple/control_dependencyConvNet/dense_6/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/ConvNet/dense_6/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_6/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/ConvNet/dense_6/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_6/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_6/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_6/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_6/Relu_grad/ReluGrad8^gradients/ConvNet/dense_6/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_6/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients/ConvNet/dense_6/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_6/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_6/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*E
_class;
97loc:@gradients/ConvNet/dense_6/BiasAdd_grad/BiasAddGrad
í
,gradients/ConvNet/dense_6/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_6/BiasAdd_grad/tuple/control_dependencyConvNet/dense_6/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ŕ
.gradients/ConvNet/dense_6/MatMul_grad/MatMul_1MatMulConvNet/dense_5/Relu?gradients/ConvNet/dense_6/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

6gradients/ConvNet/dense_6/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_6/MatMul_grad/MatMul/^gradients/ConvNet/dense_6/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_6/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_6/MatMul_grad/MatMul7^gradients/ConvNet/dense_6/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_6/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/ConvNet/dense_6/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_6/MatMul_grad/MatMul_17^gradients/ConvNet/dense_6/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*A
_class7
53loc:@gradients/ConvNet/dense_6/MatMul_grad/MatMul_1
Á
,gradients/ConvNet/dense_5/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_6/MatMul_grad/tuple/control_dependencyConvNet/dense_5/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
2gradients/ConvNet/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/ConvNet/dense_5/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_5/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_5/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_5/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_5/Relu_grad/ReluGrad8^gradients/ConvNet/dense_5/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_5/Relu_grad/ReluGrad
¨
Agradients/ConvNet/dense_5/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_5/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_5/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ConvNet/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
,gradients/ConvNet/dense_5/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_5/BiasAdd_grad/tuple/control_dependencyConvNet/dense_5/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ŕ
.gradients/ConvNet/dense_5/MatMul_grad/MatMul_1MatMulConvNet/dense_4/Relu?gradients/ConvNet/dense_5/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

6gradients/ConvNet/dense_5/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_5/MatMul_grad/MatMul/^gradients/ConvNet/dense_5/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_5/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_5/MatMul_grad/MatMul7^gradients/ConvNet/dense_5/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_5/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/ConvNet/dense_5/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_5/MatMul_grad/MatMul_17^gradients/ConvNet/dense_5/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/ConvNet/dense_5/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/ConvNet/dense_4/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_5/MatMul_grad/tuple/control_dependencyConvNet/dense_4/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/ConvNet/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_4/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Ł
7gradients/ConvNet/dense_4/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_4/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_4/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_4/Relu_grad/ReluGrad8^gradients/ConvNet/dense_4/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_4/Relu_grad/ReluGrad
¨
Agradients/ConvNet/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_4/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_4/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ConvNet/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
,gradients/ConvNet/dense_4/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_4/BiasAdd_grad/tuple/control_dependencyConvNet/dense_4/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ŕ
.gradients/ConvNet/dense_4/MatMul_grad/MatMul_1MatMulConvNet/dense_3/Relu?gradients/ConvNet/dense_4/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

6gradients/ConvNet/dense_4/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_4/MatMul_grad/MatMul/^gradients/ConvNet/dense_4/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_4/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_4/MatMul_grad/MatMul7^gradients/ConvNet/dense_4/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_4/MatMul_grad/MatMul
Ł
@gradients/ConvNet/dense_4/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_4/MatMul_grad/MatMul_17^gradients/ConvNet/dense_4/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/ConvNet/dense_4/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/ConvNet/dense_3/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_4/MatMul_grad/tuple/control_dependencyConvNet/dense_3/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
2gradients/ConvNet/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ł
7gradients/ConvNet/dense_3/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_3/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_3/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_3/Relu_grad/ReluGrad8^gradients/ConvNet/dense_3/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_3/Relu_grad/ReluGrad
¨
Agradients/ConvNet/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_3/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*E
_class;
97loc:@gradients/ConvNet/dense_3/BiasAdd_grad/BiasAddGrad
í
,gradients/ConvNet/dense_3/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_3/BiasAdd_grad/tuple/control_dependencyConvNet/dense_3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ŕ
.gradients/ConvNet/dense_3/MatMul_grad/MatMul_1MatMulConvNet/dense_2/Relu?gradients/ConvNet/dense_3/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

6gradients/ConvNet/dense_3/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_3/MatMul_grad/MatMul/^gradients/ConvNet/dense_3/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_3/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_3/MatMul_grad/MatMul7^gradients/ConvNet/dense_3/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_3/MatMul_grad/MatMul
Ł
@gradients/ConvNet/dense_3/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_3/MatMul_grad/MatMul_17^gradients/ConvNet/dense_3/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*A
_class7
53loc:@gradients/ConvNet/dense_3/MatMul_grad/MatMul_1
Á
,gradients/ConvNet/dense_2/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_3/MatMul_grad/tuple/control_dependencyConvNet/dense_2/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/ConvNet/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/ConvNet/dense_2/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_2/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_2/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_2/Relu_grad/ReluGrad8^gradients/ConvNet/dense_2/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_2/Relu_grad/ReluGrad
¨
Agradients/ConvNet/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_2/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_2/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/ConvNet/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
í
,gradients/ConvNet/dense_2/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_2/BiasAdd_grad/tuple/control_dependencyConvNet/dense_2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ŕ
.gradients/ConvNet/dense_2/MatMul_grad/MatMul_1MatMulConvNet/dense_1/Relu?gradients/ConvNet/dense_2/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

6gradients/ConvNet/dense_2/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_2/MatMul_grad/MatMul/^gradients/ConvNet/dense_2/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_2/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_2/MatMul_grad/MatMul7^gradients/ConvNet/dense_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/ConvNet/dense_2/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_2/MatMul_grad/MatMul_17^gradients/ConvNet/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/ConvNet/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/ConvNet/dense_1/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_2/MatMul_grad/tuple/control_dependencyConvNet/dense_1/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/ConvNet/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/ConvNet/dense_1/BiasAdd_grad/tuple/group_depsNoOp3^gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGrad-^gradients/ConvNet/dense_1/Relu_grad/ReluGrad
§
?gradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_1/Relu_grad/ReluGrad8^gradients/ConvNet/dense_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense_1/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGrad8^gradients/ConvNet/dense_1/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ConvNet/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
,gradients/ConvNet/dense_1/MatMul_grad/MatMulMatMul?gradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependencyConvNet/dense_1/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ţ
.gradients/ConvNet/dense_1/MatMul_grad/MatMul_1MatMulConvNet/dense/Relu?gradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

6gradients/ConvNet/dense_1/MatMul_grad/tuple/group_depsNoOp-^gradients/ConvNet/dense_1/MatMul_grad/MatMul/^gradients/ConvNet/dense_1/MatMul_grad/MatMul_1
Ľ
>gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependencyIdentity,gradients/ConvNet/dense_1/MatMul_grad/MatMul7^gradients/ConvNet/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/ConvNet/dense_1/MatMul_grad/MatMul
Ł
@gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependency_1Identity.gradients/ConvNet/dense_1/MatMul_grad/MatMul_17^gradients/ConvNet/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/ConvNet/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

˝
*gradients/ConvNet/dense/Relu_grad/ReluGradReluGrad>gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependencyConvNet/dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
0gradients/ConvNet/dense/BiasAdd_grad/BiasAddGradBiasAddGrad*gradients/ConvNet/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

5gradients/ConvNet/dense/BiasAdd_grad/tuple/group_depsNoOp1^gradients/ConvNet/dense/BiasAdd_grad/BiasAddGrad+^gradients/ConvNet/dense/Relu_grad/ReluGrad

=gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependencyIdentity*gradients/ConvNet/dense/Relu_grad/ReluGrad6^gradients/ConvNet/dense/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/ConvNet/dense/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
?gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependency_1Identity0gradients/ConvNet/dense/BiasAdd_grad/BiasAddGrad6^gradients/ConvNet/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*C
_class9
75loc:@gradients/ConvNet/dense/BiasAdd_grad/BiasAddGrad
ç
*gradients/ConvNet/dense/MatMul_grad/MatMulMatMul=gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependencyConvNet/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ç
,gradients/ConvNet/dense/MatMul_grad/MatMul_1MatMulConvNet/Flatten/flatten/Reshape=gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(

4gradients/ConvNet/dense/MatMul_grad/tuple/group_depsNoOp+^gradients/ConvNet/dense/MatMul_grad/MatMul-^gradients/ConvNet/dense/MatMul_grad/MatMul_1

<gradients/ConvNet/dense/MatMul_grad/tuple/control_dependencyIdentity*gradients/ConvNet/dense/MatMul_grad/MatMul5^gradients/ConvNet/dense/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/ConvNet/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients/ConvNet/dense/MatMul_grad/tuple/control_dependency_1Identity,gradients/ConvNet/dense/MatMul_grad/MatMul_15^gradients/ConvNet/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/ConvNet/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

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
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container 
ž
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
 *wž?*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container 
ž
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(
z
beta2_power/readIdentitybeta2_power*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
: *
T0
ľ
;ConvNet/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@ConvNet/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:

1ConvNet/dense/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *'
_class
loc:@ConvNet/dense/kernel*
valueB
 *    *
dtype0

+ConvNet/dense/kernel/Adam/Initializer/zerosFill;ConvNet/dense/kernel/Adam/Initializer/zeros/shape_as_tensor1ConvNet/dense/kernel/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@ConvNet/dense/kernel*

index_type0* 
_output_shapes
:

ş
ConvNet/dense/kernel/Adam
VariableV2* 
_output_shapes
:
*
shared_name *'
_class
loc:@ConvNet/dense/kernel*
	container *
shape:
*
dtype0
ď
 ConvNet/dense/kernel/Adam/AssignAssignConvNet/dense/kernel/Adam+ConvNet/dense/kernel/Adam/Initializer/zeros*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

ConvNet/dense/kernel/Adam/readIdentityConvNet/dense/kernel/Adam*
T0*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:

ˇ
=ConvNet/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@ConvNet/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ą
3ConvNet/dense/kernel/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@ConvNet/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-ConvNet/dense/kernel/Adam_1/Initializer/zerosFill=ConvNet/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor3ConvNet/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@ConvNet/dense/kernel*

index_type0* 
_output_shapes
:

ź
ConvNet/dense/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *'
_class
loc:@ConvNet/dense/kernel*
	container 
ő
"ConvNet/dense/kernel/Adam_1/AssignAssignConvNet/dense/kernel/Adam_1-ConvNet/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense/kernel/Adam_1/readIdentityConvNet/dense/kernel/Adam_1*'
_class
loc:@ConvNet/dense/kernel* 
_output_shapes
:
*
T0

)ConvNet/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@ConvNet/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ź
ConvNet/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *%
_class
loc:@ConvNet/dense/bias*
	container *
shape:
â
ConvNet/dense/bias/Adam/AssignAssignConvNet/dense/bias/Adam)ConvNet/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense/bias/Adam/readIdentityConvNet/dense/bias/Adam*
T0*%
_class
loc:@ConvNet/dense/bias*
_output_shapes	
:
Ą
+ConvNet/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*%
_class
loc:@ConvNet/dense/bias*
valueB*    *
dtype0
Ž
ConvNet/dense/bias/Adam_1
VariableV2*
shared_name *%
_class
loc:@ConvNet/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
č
 ConvNet/dense/bias/Adam_1/AssignAssignConvNet/dense/bias/Adam_1+ConvNet/dense/bias/Adam_1/Initializer/zeros*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

ConvNet/dense/bias/Adam_1/readIdentityConvNet/dense/bias/Adam_1*
T0*%
_class
loc:@ConvNet/dense/bias*
_output_shapes	
:
š
=ConvNet/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ł
3ConvNet/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-ConvNet/dense_1/kernel/Adam/Initializer/zerosFill=ConvNet/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_1/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@ConvNet/dense_1/kernel*

index_type0* 
_output_shapes
:
*
T0
ž
ConvNet/dense_1/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_1/kernel*
	container *
shape:

÷
"ConvNet/dense_1/kernel/Adam/AssignAssignConvNet/dense_1/kernel/Adam-ConvNet/dense_1/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(

 ConvNet/dense_1/kernel/Adam/readIdentityConvNet/dense_1/kernel/Adam*
T0*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:

ť
?ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_1/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_1/kernel*

index_type0
Ŕ
ConvNet/dense_1/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_1/kernel*
	container 
ý
$ConvNet/dense_1/kernel/Adam_1/AssignAssignConvNet/dense_1/kernel/Adam_1/ConvNet/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel
Ł
"ConvNet/dense_1/kernel/Adam_1/readIdentityConvNet/dense_1/kernel/Adam_1*)
_class
loc:@ConvNet/dense_1/kernel* 
_output_shapes
:
*
T0
Ł
+ConvNet/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
°
ConvNet/dense_1/bias/Adam
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
 ConvNet/dense_1/bias/Adam/AssignAssignConvNet/dense_1/bias/Adam+ConvNet/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_1/bias/Adam/readIdentityConvNet/dense_1/bias/Adam*
T0*'
_class
loc:@ConvNet/dense_1/bias*
_output_shapes	
:
Ľ
-ConvNet/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_1/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
đ
"ConvNet/dense_1/bias/Adam_1/AssignAssignConvNet/dense_1/bias/Adam_1-ConvNet/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_1/bias/Adam_1/readIdentityConvNet/dense_1/bias/Adam_1*
T0*'
_class
loc:@ConvNet/dense_1/bias*
_output_shapes	
:
š
=ConvNet/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ł
3ConvNet/dense_2/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-ConvNet/dense_2/kernel/Adam/Initializer/zerosFill=ConvNet/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_2/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_2/kernel*

index_type0* 
_output_shapes
:

ž
ConvNet/dense_2/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_2/kernel*
	container 
÷
"ConvNet/dense_2/kernel/Adam/AssignAssignConvNet/dense_2/kernel/Adam-ConvNet/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_2/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense_2/kernel/Adam/readIdentityConvNet/dense_2/kernel/Adam* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_2/kernel
ť
?ConvNet/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*)
_class
loc:@ConvNet/dense_2/kernel*
valueB"      
Ľ
5ConvNet/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_2/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_2/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_2/kernel*

index_type0
Ŕ
ConvNet/dense_2/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_2/kernel*
	container *
shape:

ý
$ConvNet/dense_2/kernel/Adam_1/AssignAssignConvNet/dense_2/kernel/Adam_1/ConvNet/dense_2/kernel/Adam_1/Initializer/zeros*
T0*)
_class
loc:@ConvNet/dense_2/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ł
"ConvNet/dense_2/kernel/Adam_1/readIdentityConvNet/dense_2/kernel/Adam_1*)
_class
loc:@ConvNet/dense_2/kernel* 
_output_shapes
:
*
T0
Ł
+ConvNet/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
°
ConvNet/dense_2/bias/Adam
VariableV2*'
_class
loc:@ConvNet/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ę
 ConvNet/dense_2/bias/Adam/AssignAssignConvNet/dense_2/bias/Adam+ConvNet/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_2/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_2/bias/Adam/readIdentityConvNet/dense_2/bias/Adam*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_2/bias
Ľ
-ConvNet/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_2/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_2/bias/Adam_1
VariableV2*'
_class
loc:@ConvNet/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
đ
"ConvNet/dense_2/bias/Adam_1/AssignAssignConvNet/dense_2/bias/Adam_1-ConvNet/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_2/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_2/bias/Adam_1/readIdentityConvNet/dense_2/bias/Adam_1*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_2/bias
š
=ConvNet/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ł
3ConvNet/dense_3/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@ConvNet/dense_3/kernel*
valueB
 *    

-ConvNet/dense_3/kernel/Adam/Initializer/zerosFill=ConvNet/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_3/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_3/kernel*

index_type0* 
_output_shapes
:

ž
ConvNet/dense_3/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_3/kernel*
	container 
÷
"ConvNet/dense_3/kernel/Adam/AssignAssignConvNet/dense_3/kernel/Adam-ConvNet/dense_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_3/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense_3/kernel/Adam/readIdentityConvNet/dense_3/kernel/Adam*)
_class
loc:@ConvNet/dense_3/kernel* 
_output_shapes
:
*
T0
ť
?ConvNet/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_3/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_3/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_3/kernel*

index_type0
Ŕ
ConvNet/dense_3/kernel/Adam_1
VariableV2*)
_class
loc:@ConvNet/dense_3/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ý
$ConvNet/dense_3/kernel/Adam_1/AssignAssignConvNet/dense_3/kernel/Adam_1/ConvNet/dense_3/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_3/kernel*
validate_shape(
Ł
"ConvNet/dense_3/kernel/Adam_1/readIdentityConvNet/dense_3/kernel/Adam_1*
T0*)
_class
loc:@ConvNet/dense_3/kernel* 
_output_shapes
:

Ł
+ConvNet/dense_3/bias/Adam/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
°
ConvNet/dense_3/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_3/bias*
	container *
shape:
ę
 ConvNet/dense_3/bias/Adam/AssignAssignConvNet/dense_3/bias/Adam+ConvNet/dense_3/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@ConvNet/dense_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

ConvNet/dense_3/bias/Adam/readIdentityConvNet/dense_3/bias/Adam*'
_class
loc:@ConvNet/dense_3/bias*
_output_shapes	
:*
T0
Ľ
-ConvNet/dense_3/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_3/bias/Adam_1
VariableV2*'
_class
loc:@ConvNet/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
đ
"ConvNet/dense_3/bias/Adam_1/AssignAssignConvNet/dense_3/bias/Adam_1-ConvNet/dense_3/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_3/bias*
validate_shape(

 ConvNet/dense_3/bias/Adam_1/readIdentityConvNet/dense_3/bias/Adam_1*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_3/bias
š
=ConvNet/dense_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ł
3ConvNet/dense_4/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *)
_class
loc:@ConvNet/dense_4/kernel*
valueB
 *    *
dtype0

-ConvNet/dense_4/kernel/Adam/Initializer/zerosFill=ConvNet/dense_4/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_4/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_4/kernel*

index_type0* 
_output_shapes
:

ž
ConvNet/dense_4/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_4/kernel*
	container 
÷
"ConvNet/dense_4/kernel/Adam/AssignAssignConvNet/dense_4/kernel/Adam-ConvNet/dense_4/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_4/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense_4/kernel/Adam/readIdentityConvNet/dense_4/kernel/Adam*
T0*)
_class
loc:@ConvNet/dense_4/kernel* 
_output_shapes
:

ť
?ConvNet/dense_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_4/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_4/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_4/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_4/kernel*

index_type0* 
_output_shapes
:

Ŕ
ConvNet/dense_4/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_4/kernel*
	container *
shape:

ý
$ConvNet/dense_4/kernel/Adam_1/AssignAssignConvNet/dense_4/kernel/Adam_1/ConvNet/dense_4/kernel/Adam_1/Initializer/zeros*
T0*)
_class
loc:@ConvNet/dense_4/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ł
"ConvNet/dense_4/kernel/Adam_1/readIdentityConvNet/dense_4/kernel/Adam_1* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_4/kernel
Ł
+ConvNet/dense_4/bias/Adam/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_4/bias*
valueB*    *
dtype0*
_output_shapes	
:
°
ConvNet/dense_4/bias/Adam
VariableV2*
shared_name *'
_class
loc:@ConvNet/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
 ConvNet/dense_4/bias/Adam/AssignAssignConvNet/dense_4/bias/Adam+ConvNet/dense_4/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_4/bias*
validate_shape(

ConvNet/dense_4/bias/Adam/readIdentityConvNet/dense_4/bias/Adam*
T0*'
_class
loc:@ConvNet/dense_4/bias*
_output_shapes	
:
Ľ
-ConvNet/dense_4/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_4/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_4/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_4/bias
đ
"ConvNet/dense_4/bias/Adam_1/AssignAssignConvNet/dense_4/bias/Adam_1-ConvNet/dense_4/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_4/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_4/bias/Adam_1/readIdentityConvNet/dense_4/bias/Adam_1*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_4/bias
š
=ConvNet/dense_5/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_5/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ł
3ConvNet/dense_5/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-ConvNet/dense_5/kernel/Adam/Initializer/zerosFill=ConvNet/dense_5/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_5/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@ConvNet/dense_5/kernel*

index_type0* 
_output_shapes
:
*
T0
ž
ConvNet/dense_5/kernel/Adam
VariableV2*)
_class
loc:@ConvNet/dense_5/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
÷
"ConvNet/dense_5/kernel/Adam/AssignAssignConvNet/dense_5/kernel/Adam-ConvNet/dense_5/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_5/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense_5/kernel/Adam/readIdentityConvNet/dense_5/kernel/Adam*
T0*)
_class
loc:@ConvNet/dense_5/kernel* 
_output_shapes
:

ť
?ConvNet/dense_5/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*)
_class
loc:@ConvNet/dense_5/kernel*
valueB"      *
dtype0
Ľ
5ConvNet/dense_5/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_5/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_5/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_5/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_5/kernel*

index_type0* 
_output_shapes
:

Ŕ
ConvNet/dense_5/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@ConvNet/dense_5/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$ConvNet/dense_5/kernel/Adam_1/AssignAssignConvNet/dense_5/kernel/Adam_1/ConvNet/dense_5/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_5/kernel
Ł
"ConvNet/dense_5/kernel/Adam_1/readIdentityConvNet/dense_5/kernel/Adam_1* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_5/kernel
Ł
+ConvNet/dense_5/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*'
_class
loc:@ConvNet/dense_5/bias*
valueB*    
°
ConvNet/dense_5/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_5/bias*
	container *
shape:
ę
 ConvNet/dense_5/bias/Adam/AssignAssignConvNet/dense_5/bias/Adam+ConvNet/dense_5/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_5/bias

ConvNet/dense_5/bias/Adam/readIdentityConvNet/dense_5/bias/Adam*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_5/bias
Ľ
-ConvNet/dense_5/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_5/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_5/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_5/bias*
	container *
shape:
đ
"ConvNet/dense_5/bias/Adam_1/AssignAssignConvNet/dense_5/bias/Adam_1-ConvNet/dense_5/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_5/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_5/bias/Adam_1/readIdentityConvNet/dense_5/bias/Adam_1*
T0*'
_class
loc:@ConvNet/dense_5/bias*
_output_shapes	
:
š
=ConvNet/dense_6/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_6/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ł
3ConvNet/dense_6/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@ConvNet/dense_6/kernel*
valueB
 *    

-ConvNet/dense_6/kernel/Adam/Initializer/zerosFill=ConvNet/dense_6/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_6/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_6/kernel*

index_type0* 
_output_shapes
:

ž
ConvNet/dense_6/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_6/kernel*
	container *
shape:

÷
"ConvNet/dense_6/kernel/Adam/AssignAssignConvNet/dense_6/kernel/Adam-ConvNet/dense_6/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_6/kernel*
validate_shape(* 
_output_shapes
:


 ConvNet/dense_6/kernel/Adam/readIdentityConvNet/dense_6/kernel/Adam*
T0*)
_class
loc:@ConvNet/dense_6/kernel* 
_output_shapes
:

ť
?ConvNet/dense_6/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*)
_class
loc:@ConvNet/dense_6/kernel*
valueB"      *
dtype0
Ľ
5ConvNet/dense_6/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_6/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/ConvNet/dense_6/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_6/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_6/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_6/kernel*

index_type0
Ŕ
ConvNet/dense_6/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_6/kernel
ý
$ConvNet/dense_6/kernel/Adam_1/AssignAssignConvNet/dense_6/kernel/Adam_1/ConvNet/dense_6/kernel/Adam_1/Initializer/zeros*)
_class
loc:@ConvNet/dense_6/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ł
"ConvNet/dense_6/kernel/Adam_1/readIdentityConvNet/dense_6/kernel/Adam_1* 
_output_shapes
:
*
T0*)
_class
loc:@ConvNet/dense_6/kernel
Ł
+ConvNet/dense_6/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*'
_class
loc:@ConvNet/dense_6/bias*
valueB*    *
dtype0
°
ConvNet/dense_6/bias/Adam
VariableV2*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_6/bias*
	container *
shape:*
dtype0
ę
 ConvNet/dense_6/bias/Adam/AssignAssignConvNet/dense_6/bias/Adam+ConvNet/dense_6/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_6/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_6/bias/Adam/readIdentityConvNet/dense_6/bias/Adam*
T0*'
_class
loc:@ConvNet/dense_6/bias*
_output_shapes	
:
Ľ
-ConvNet/dense_6/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_6/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_6/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_6/bias*
	container *
shape:
đ
"ConvNet/dense_6/bias/Adam_1/AssignAssignConvNet/dense_6/bias/Adam_1-ConvNet/dense_6/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_6/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_6/bias/Adam_1/readIdentityConvNet/dense_6/bias/Adam_1*
T0*'
_class
loc:@ConvNet/dense_6/bias*
_output_shapes	
:
š
=ConvNet/dense_7/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*)
_class
loc:@ConvNet/dense_7/kernel*
valueB"      
Ł
3ConvNet/dense_7/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@ConvNet/dense_7/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-ConvNet/dense_7/kernel/Adam/Initializer/zerosFill=ConvNet/dense_7/kernel/Adam/Initializer/zeros/shape_as_tensor3ConvNet/dense_7/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_7/kernel*

index_type0* 
_output_shapes
:

ž
ConvNet/dense_7/kernel/Adam
VariableV2*)
_class
loc:@ConvNet/dense_7/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
÷
"ConvNet/dense_7/kernel/Adam/AssignAssignConvNet/dense_7/kernel/Adam-ConvNet/dense_7/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@ConvNet/dense_7/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

 ConvNet/dense_7/kernel/Adam/readIdentityConvNet/dense_7/kernel/Adam*
T0*)
_class
loc:@ConvNet/dense_7/kernel* 
_output_shapes
:

ť
?ConvNet/dense_7/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@ConvNet/dense_7/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ľ
5ConvNet/dense_7/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@ConvNet/dense_7/kernel*
valueB
 *    

/ConvNet/dense_7/kernel/Adam_1/Initializer/zerosFill?ConvNet/dense_7/kernel/Adam_1/Initializer/zeros/shape_as_tensor5ConvNet/dense_7/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@ConvNet/dense_7/kernel*

index_type0* 
_output_shapes
:

Ŕ
ConvNet/dense_7/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *)
_class
loc:@ConvNet/dense_7/kernel*
	container *
shape:
*
dtype0
ý
$ConvNet/dense_7/kernel/Adam_1/AssignAssignConvNet/dense_7/kernel/Adam_1/ConvNet/dense_7/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_7/kernel*
validate_shape(
Ł
"ConvNet/dense_7/kernel/Adam_1/readIdentityConvNet/dense_7/kernel/Adam_1*
T0*)
_class
loc:@ConvNet/dense_7/kernel* 
_output_shapes
:

Ł
+ConvNet/dense_7/bias/Adam/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_7/bias*
valueB*    *
dtype0*
_output_shapes	
:
°
ConvNet/dense_7/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_7/bias*
	container 
ę
 ConvNet/dense_7/bias/Adam/AssignAssignConvNet/dense_7/bias/Adam+ConvNet/dense_7/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_7/bias*
validate_shape(*
_output_shapes	
:

ConvNet/dense_7/bias/Adam/readIdentityConvNet/dense_7/bias/Adam*
_output_shapes	
:*
T0*'
_class
loc:@ConvNet/dense_7/bias
Ľ
-ConvNet/dense_7/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@ConvNet/dense_7/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
ConvNet/dense_7/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@ConvNet/dense_7/bias*
	container 
đ
"ConvNet/dense_7/bias/Adam_1/AssignAssignConvNet/dense_7/bias/Adam_1-ConvNet/dense_7/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_7/bias*
validate_shape(*
_output_shapes	
:

 ConvNet/dense_7/bias/Adam_1/readIdentityConvNet/dense_7/bias/Adam_1*'
_class
loc:@ConvNet/dense_7/bias*
_output_shapes	
:*
T0
Ç
DConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB"      *
dtype0*
_output_shapes
:
ą
:ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ź
4ConvNet/cnn_logits_out/kernel/Adam/Initializer/zerosFillDConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/shape_as_tensor:ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros/Const*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*

index_type0*
_output_shapes
:	
Ę
"ConvNet/cnn_logits_out/kernel/Adam
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
	container 

)ConvNet/cnn_logits_out/kernel/Adam/AssignAssign"ConvNet/cnn_logits_out/kernel/Adam4ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
ł
'ConvNet/cnn_logits_out/kernel/Adam/readIdentity"ConvNet/cnn_logits_out/kernel/Adam*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
_output_shapes
:	
É
FConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB"      *
dtype0*
_output_shapes
:
ł
<ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
˛
6ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zerosFillFConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/shape_as_tensor<ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*

index_type0
Ě
$ConvNet/cnn_logits_out/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
	container *
shape:	

+ConvNet/cnn_logits_out/kernel/Adam_1/AssignAssign$ConvNet/cnn_logits_out/kernel/Adam_16ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
ˇ
)ConvNet/cnn_logits_out/kernel/Adam_1/readIdentity$ConvNet/cnn_logits_out/kernel/Adam_1*
_output_shapes
:	*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel
Ż
2ConvNet/cnn_logits_out/bias/Adam/Initializer/zerosConst*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
valueB*    *
dtype0*
_output_shapes
:
ź
 ConvNet/cnn_logits_out/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape:

'ConvNet/cnn_logits_out/bias/Adam/AssignAssign ConvNet/cnn_logits_out/bias/Adam2ConvNet/cnn_logits_out/bias/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:
¨
%ConvNet/cnn_logits_out/bias/Adam/readIdentity ConvNet/cnn_logits_out/bias/Adam*
_output_shapes
:*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias
ą
4ConvNet/cnn_logits_out/bias/Adam_1/Initializer/zerosConst*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
valueB*    *
dtype0*
_output_shapes
:
ž
"ConvNet/cnn_logits_out/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
	container *
shape:

)ConvNet/cnn_logits_out/bias/Adam_1/AssignAssign"ConvNet/cnn_logits_out/bias/Adam_14ConvNet/cnn_logits_out/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:
Ź
'ConvNet/cnn_logits_out/bias/Adam_1/readIdentity"ConvNet/cnn_logits_out/bias/Adam_1*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
:
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o:*
dtype0
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
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

*Adam/update_ConvNet/dense/kernel/ApplyAdam	ApplyAdamConvNet/dense/kernelConvNet/dense/kernel/AdamConvNet/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/ConvNet/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@ConvNet/dense/kernel*
use_nesterov( * 
_output_shapes
:


(Adam/update_ConvNet/dense/bias/ApplyAdam	ApplyAdamConvNet/dense/biasConvNet/dense/bias/AdamConvNet/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon?gradients/ConvNet/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*%
_class
loc:@ConvNet/dense/bias*
use_nesterov( 
Ş
,Adam/update_ConvNet/dense_1/kernel/ApplyAdam	ApplyAdamConvNet/dense_1/kernelConvNet/dense_1/kernel/AdamConvNet/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*)
_class
loc:@ConvNet/dense_1/kernel

*Adam/update_ConvNet/dense_1/bias/ApplyAdam	ApplyAdamConvNet/dense_1/biasConvNet/dense_1/bias/AdamConvNet/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@ConvNet/dense_1/bias*
use_nesterov( *
_output_shapes	
:
Ş
,Adam/update_ConvNet/dense_2/kernel/ApplyAdam	ApplyAdamConvNet/dense_2/kernelConvNet/dense_2/kernel/AdamConvNet/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ConvNet/dense_2/kernel*
use_nesterov( * 
_output_shapes
:


*Adam/update_ConvNet/dense_2/bias/ApplyAdam	ApplyAdamConvNet/dense_2/biasConvNet/dense_2/bias/AdamConvNet/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@ConvNet/dense_2/bias*
use_nesterov( *
_output_shapes	
:
Ş
,Adam/update_ConvNet/dense_3/kernel/ApplyAdam	ApplyAdamConvNet/dense_3/kernelConvNet/dense_3/kernel/AdamConvNet/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_3/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*)
_class
loc:@ConvNet/dense_3/kernel

*Adam/update_ConvNet/dense_3/bias/ApplyAdam	ApplyAdamConvNet/dense_3/biasConvNet/dense_3/bias/AdamConvNet/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@ConvNet/dense_3/bias*
use_nesterov( *
_output_shapes	
:
Ş
,Adam/update_ConvNet/dense_4/kernel/ApplyAdam	ApplyAdamConvNet/dense_4/kernelConvNet/dense_4/kernel/AdamConvNet/dense_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ConvNet/dense_4/kernel*
use_nesterov( * 
_output_shapes
:


*Adam/update_ConvNet/dense_4/bias/ApplyAdam	ApplyAdamConvNet/dense_4/biasConvNet/dense_4/bias/AdamConvNet/dense_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_4/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@ConvNet/dense_4/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ş
,Adam/update_ConvNet/dense_5/kernel/ApplyAdam	ApplyAdamConvNet/dense_5/kernelConvNet/dense_5/kernel/AdamConvNet/dense_5/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_5/MatMul_grad/tuple/control_dependency_1*)
_class
loc:@ConvNet/dense_5/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0

*Adam/update_ConvNet/dense_5/bias/ApplyAdam	ApplyAdamConvNet/dense_5/biasConvNet/dense_5/bias/AdamConvNet/dense_5/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_5/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@ConvNet/dense_5/bias*
use_nesterov( *
_output_shapes	
:
Ş
,Adam/update_ConvNet/dense_6/kernel/ApplyAdam	ApplyAdamConvNet/dense_6/kernelConvNet/dense_6/kernel/AdamConvNet/dense_6/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_6/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@ConvNet/dense_6/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 

*Adam/update_ConvNet/dense_6/bias/ApplyAdam	ApplyAdamConvNet/dense_6/biasConvNet/dense_6/bias/AdamConvNet/dense_6/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_6/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@ConvNet/dense_6/bias*
use_nesterov( *
_output_shapes	
:
Ş
,Adam/update_ConvNet/dense_7/kernel/ApplyAdam	ApplyAdamConvNet/dense_7/kernelConvNet/dense_7/kernel/AdamConvNet/dense_7/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/ConvNet/dense_7/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@ConvNet/dense_7/kernel*
use_nesterov( * 
_output_shapes
:


*Adam/update_ConvNet/dense_7/bias/ApplyAdam	ApplyAdamConvNet/dense_7/biasConvNet/dense_7/bias/AdamConvNet/dense_7/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/ConvNet/dense_7/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@ConvNet/dense_7/bias*
use_nesterov( 
Ó
3Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam	ApplyAdamConvNet/cnn_logits_out/kernel"ConvNet/cnn_logits_out/kernel/Adam$ConvNet/cnn_logits_out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/ConvNet/cnn_logits_out/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
use_nesterov( *
_output_shapes
:	
Ĺ
1Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam	ApplyAdamConvNet/cnn_logits_out/bias ConvNet/cnn_logits_out/bias/Adam"ConvNet/cnn_logits_out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/ConvNet/cnn_logits_out/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
use_nesterov( 
Ä
Adam/mulMulbeta1_power/read
Adam/beta12^Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam4^Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam)^Adam/update_ConvNet/dense/bias/ApplyAdam+^Adam/update_ConvNet/dense/kernel/ApplyAdam+^Adam/update_ConvNet/dense_1/bias/ApplyAdam-^Adam/update_ConvNet/dense_1/kernel/ApplyAdam+^Adam/update_ConvNet/dense_2/bias/ApplyAdam-^Adam/update_ConvNet/dense_2/kernel/ApplyAdam+^Adam/update_ConvNet/dense_3/bias/ApplyAdam-^Adam/update_ConvNet/dense_3/kernel/ApplyAdam+^Adam/update_ConvNet/dense_4/bias/ApplyAdam-^Adam/update_ConvNet/dense_4/kernel/ApplyAdam+^Adam/update_ConvNet/dense_5/bias/ApplyAdam-^Adam/update_ConvNet/dense_5/kernel/ApplyAdam+^Adam/update_ConvNet/dense_6/bias/ApplyAdam-^Adam/update_ConvNet/dense_6/kernel/ApplyAdam+^Adam/update_ConvNet/dense_7/bias/ApplyAdam-^Adam/update_ConvNet/dense_7/kernel/ApplyAdam*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
_output_shapes
: *
T0
Ś
Adam/AssignAssignbeta1_powerAdam/mul*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
Ć

Adam/mul_1Mulbeta2_power/read
Adam/beta22^Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam4^Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam)^Adam/update_ConvNet/dense/bias/ApplyAdam+^Adam/update_ConvNet/dense/kernel/ApplyAdam+^Adam/update_ConvNet/dense_1/bias/ApplyAdam-^Adam/update_ConvNet/dense_1/kernel/ApplyAdam+^Adam/update_ConvNet/dense_2/bias/ApplyAdam-^Adam/update_ConvNet/dense_2/kernel/ApplyAdam+^Adam/update_ConvNet/dense_3/bias/ApplyAdam-^Adam/update_ConvNet/dense_3/kernel/ApplyAdam+^Adam/update_ConvNet/dense_4/bias/ApplyAdam-^Adam/update_ConvNet/dense_4/kernel/ApplyAdam+^Adam/update_ConvNet/dense_5/bias/ApplyAdam-^Adam/update_ConvNet/dense_5/kernel/ApplyAdam+^Adam/update_ConvNet/dense_6/bias/ApplyAdam-^Adam/update_ConvNet/dense_6/kernel/ApplyAdam+^Adam/update_ConvNet/dense_7/bias/ApplyAdam-^Adam/update_ConvNet/dense_7/kernel/ApplyAdam*
_output_shapes
: *
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias
Ş
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
đ
AdamNoOp^Adam/Assign^Adam/Assign_12^Adam/update_ConvNet/cnn_logits_out/bias/ApplyAdam4^Adam/update_ConvNet/cnn_logits_out/kernel/ApplyAdam)^Adam/update_ConvNet/dense/bias/ApplyAdam+^Adam/update_ConvNet/dense/kernel/ApplyAdam+^Adam/update_ConvNet/dense_1/bias/ApplyAdam-^Adam/update_ConvNet/dense_1/kernel/ApplyAdam+^Adam/update_ConvNet/dense_2/bias/ApplyAdam-^Adam/update_ConvNet/dense_2/kernel/ApplyAdam+^Adam/update_ConvNet/dense_3/bias/ApplyAdam-^Adam/update_ConvNet/dense_3/kernel/ApplyAdam+^Adam/update_ConvNet/dense_4/bias/ApplyAdam-^Adam/update_ConvNet/dense_4/kernel/ApplyAdam+^Adam/update_ConvNet/dense_5/bias/ApplyAdam-^Adam/update_ConvNet/dense_5/kernel/ApplyAdam+^Adam/update_ConvNet/dense_6/bias/ApplyAdam-^Adam/update_ConvNet/dense_6/kernel/ApplyAdam+^Adam/update_ConvNet/dense_7/bias/ApplyAdam-^Adam/update_ConvNet/dense_7/kernel/ApplyAdam
i
soft_predictSoftmaxConvNet/cnn_logits_out/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
Variable/AssignAssignVariablezeros*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:
Č
initNoOp(^ConvNet/cnn_logits_out/bias/Adam/Assign*^ConvNet/cnn_logits_out/bias/Adam_1/Assign#^ConvNet/cnn_logits_out/bias/Assign*^ConvNet/cnn_logits_out/kernel/Adam/Assign,^ConvNet/cnn_logits_out/kernel/Adam_1/Assign%^ConvNet/cnn_logits_out/kernel/Assign^ConvNet/dense/bias/Adam/Assign!^ConvNet/dense/bias/Adam_1/Assign^ConvNet/dense/bias/Assign!^ConvNet/dense/kernel/Adam/Assign#^ConvNet/dense/kernel/Adam_1/Assign^ConvNet/dense/kernel/Assign!^ConvNet/dense_1/bias/Adam/Assign#^ConvNet/dense_1/bias/Adam_1/Assign^ConvNet/dense_1/bias/Assign#^ConvNet/dense_1/kernel/Adam/Assign%^ConvNet/dense_1/kernel/Adam_1/Assign^ConvNet/dense_1/kernel/Assign!^ConvNet/dense_2/bias/Adam/Assign#^ConvNet/dense_2/bias/Adam_1/Assign^ConvNet/dense_2/bias/Assign#^ConvNet/dense_2/kernel/Adam/Assign%^ConvNet/dense_2/kernel/Adam_1/Assign^ConvNet/dense_2/kernel/Assign!^ConvNet/dense_3/bias/Adam/Assign#^ConvNet/dense_3/bias/Adam_1/Assign^ConvNet/dense_3/bias/Assign#^ConvNet/dense_3/kernel/Adam/Assign%^ConvNet/dense_3/kernel/Adam_1/Assign^ConvNet/dense_3/kernel/Assign!^ConvNet/dense_4/bias/Adam/Assign#^ConvNet/dense_4/bias/Adam_1/Assign^ConvNet/dense_4/bias/Assign#^ConvNet/dense_4/kernel/Adam/Assign%^ConvNet/dense_4/kernel/Adam_1/Assign^ConvNet/dense_4/kernel/Assign!^ConvNet/dense_5/bias/Adam/Assign#^ConvNet/dense_5/bias/Adam_1/Assign^ConvNet/dense_5/bias/Assign#^ConvNet/dense_5/kernel/Adam/Assign%^ConvNet/dense_5/kernel/Adam_1/Assign^ConvNet/dense_5/kernel/Assign!^ConvNet/dense_6/bias/Adam/Assign#^ConvNet/dense_6/bias/Adam_1/Assign^ConvNet/dense_6/bias/Assign#^ConvNet/dense_6/kernel/Adam/Assign%^ConvNet/dense_6/kernel/Adam_1/Assign^ConvNet/dense_6/kernel/Assign!^ConvNet/dense_7/bias/Adam/Assign#^ConvNet/dense_7/bias/Adam_1/Assign^ConvNet/dense_7/bias/Assign#^ConvNet/dense_7/kernel/Adam/Assign%^ConvNet/dense_7/kernel/Adam_1/Assign^ConvNet/dense_7/kernel/Assign^Variable/Assign^beta1_power/Assign^beta2_power/Assign

optimizationsConst*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion*
dtype0*
_output_shapes
:
¨

IteratorV2
IteratorV2*
_output_shapes
: *
	container *
output_types
2	*
shared_name *:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

TensorSliceDatasetTensorSliceDatasetflat_filenames*
_output_shapes
: *
output_shapes
: *
Toutput_types
2*
_class
loc:@IteratorV2
Ö
FlatMapDatasetFlatMapDatasetTensorSliceDataset*
_class
loc:@IteratorV2*)
f$R"
 Dataset_flat_map_read_one_file_4*
output_types
2*

Targuments
 *
_output_shapes
: *
output_shapes
: 


MapDataset
MapDatasetFlatMapDataset* 
output_shapes
::*
_class
loc:@IteratorV2*)
f$R"
 Dataset_map_transform_to_orig_10*
use_inter_op_parallelism(*
output_types
2	*

Targuments
 *
_output_shapes
: *
preserve_cardinality( 
×
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
×
BatchDatasetV2BatchDatasetV2ShuffleDataset
batch_sizedrop_remainder*
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2*
_output_shapes
: 
Ě
PrefetchDatasetPrefetchDatasetBatchDatasetV2buffer_size_1*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2	
Í
OptimizeDatasetOptimizeDatasetPrefetchDatasetoptimizations*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2	
¸
ModelDatasetModelDatasetOptimizeDataset*
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2*
_output_shapes
: 
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
ż
IteratorGetNextIteratorGetNext
IteratorV2*;
_output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

optimizations_1Const*
_output_shapes
:*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion*
dtype0
Ş
IteratorV2_1
IteratorV2*
output_types
2	*
shared_name *:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
	container 
¤
TensorSliceDataset_1TensorSliceDatasetflat_filenames_1*
_output_shapes
: *
output_shapes
: *
Toutput_types
2*
_class
loc:@IteratorV2_1
Ý
FlatMapDataset_1FlatMapDatasetTensorSliceDataset_1*
_output_shapes
: *
output_shapes
: *
_class
loc:@IteratorV2_1**
f%R#
!Dataset_flat_map_read_one_file_36*
output_types
2*

Targuments
 

MapDataset_1
MapDatasetFlatMapDataset_1* 
output_shapes
::*
_class
loc:@IteratorV2_1*)
f$R"
 Dataset_map_transform_to_orig_42*
output_types
2	*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( 
Ý
BatchDatasetV2_1BatchDatasetV2MapDataset_1batch_size_1drop_remainder_1*
_output_shapes
: *
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2_1
Ň
PrefetchDataset_1PrefetchDatasetBatchDatasetV2_1buffer_size_2*
_output_shapes
: *
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2_1
Ő
OptimizeDataset_1OptimizeDatasetPrefetchDataset_1optimizations_1*
_output_shapes
: *
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2_1
ž
ModelDataset_1ModelDatasetOptimizeDataset_1*
_output_shapes
: *
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
_class
loc:@IteratorV2_1
]
MakeIterator_1MakeIteratorModelDataset_1IteratorV2_1*
_class
loc:@IteratorV2_1
X
IteratorToStringHandle_1IteratorToStringHandleIteratorV2_1*
_output_shapes
: 
Ă
IteratorGetNext_1IteratorGetNextIteratorV2_1*;
_output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
output_types
2	*:
output_shapes)
':˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
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
shape: *
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_5cbb022859144ee892aa9d0744a67e9b/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
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
ĺ
save/SaveV2/tensor_namesConst"/device:CPU:0*
value˙Bü9BConvNet/cnn_logits_out/biasB ConvNet/cnn_logits_out/bias/AdamB"ConvNet/cnn_logits_out/bias/Adam_1BConvNet/cnn_logits_out/kernelB"ConvNet/cnn_logits_out/kernel/AdamB$ConvNet/cnn_logits_out/kernel/Adam_1BConvNet/dense/biasBConvNet/dense/bias/AdamBConvNet/dense/bias/Adam_1BConvNet/dense/kernelBConvNet/dense/kernel/AdamBConvNet/dense/kernel/Adam_1BConvNet/dense_1/biasBConvNet/dense_1/bias/AdamBConvNet/dense_1/bias/Adam_1BConvNet/dense_1/kernelBConvNet/dense_1/kernel/AdamBConvNet/dense_1/kernel/Adam_1BConvNet/dense_2/biasBConvNet/dense_2/bias/AdamBConvNet/dense_2/bias/Adam_1BConvNet/dense_2/kernelBConvNet/dense_2/kernel/AdamBConvNet/dense_2/kernel/Adam_1BConvNet/dense_3/biasBConvNet/dense_3/bias/AdamBConvNet/dense_3/bias/Adam_1BConvNet/dense_3/kernelBConvNet/dense_3/kernel/AdamBConvNet/dense_3/kernel/Adam_1BConvNet/dense_4/biasBConvNet/dense_4/bias/AdamBConvNet/dense_4/bias/Adam_1BConvNet/dense_4/kernelBConvNet/dense_4/kernel/AdamBConvNet/dense_4/kernel/Adam_1BConvNet/dense_5/biasBConvNet/dense_5/bias/AdamBConvNet/dense_5/bias/Adam_1BConvNet/dense_5/kernelBConvNet/dense_5/kernel/AdamBConvNet/dense_5/kernel/Adam_1BConvNet/dense_6/biasBConvNet/dense_6/bias/AdamBConvNet/dense_6/bias/Adam_1BConvNet/dense_6/kernelBConvNet/dense_6/kernel/AdamBConvNet/dense_6/kernel/Adam_1BConvNet/dense_7/biasBConvNet/dense_7/bias/AdamBConvNet/dense_7/bias/Adam_1BConvNet/dense_7/kernelBConvNet/dense_7/kernel/AdamBConvNet/dense_7/kernel/Adam_1BVariableBbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:9
ĺ
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:9
Ż
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConvNet/cnn_logits_out/bias ConvNet/cnn_logits_out/bias/Adam"ConvNet/cnn_logits_out/bias/Adam_1ConvNet/cnn_logits_out/kernel"ConvNet/cnn_logits_out/kernel/Adam$ConvNet/cnn_logits_out/kernel/Adam_1ConvNet/dense/biasConvNet/dense/bias/AdamConvNet/dense/bias/Adam_1ConvNet/dense/kernelConvNet/dense/kernel/AdamConvNet/dense/kernel/Adam_1ConvNet/dense_1/biasConvNet/dense_1/bias/AdamConvNet/dense_1/bias/Adam_1ConvNet/dense_1/kernelConvNet/dense_1/kernel/AdamConvNet/dense_1/kernel/Adam_1ConvNet/dense_2/biasConvNet/dense_2/bias/AdamConvNet/dense_2/bias/Adam_1ConvNet/dense_2/kernelConvNet/dense_2/kernel/AdamConvNet/dense_2/kernel/Adam_1ConvNet/dense_3/biasConvNet/dense_3/bias/AdamConvNet/dense_3/bias/Adam_1ConvNet/dense_3/kernelConvNet/dense_3/kernel/AdamConvNet/dense_3/kernel/Adam_1ConvNet/dense_4/biasConvNet/dense_4/bias/AdamConvNet/dense_4/bias/Adam_1ConvNet/dense_4/kernelConvNet/dense_4/kernel/AdamConvNet/dense_4/kernel/Adam_1ConvNet/dense_5/biasConvNet/dense_5/bias/AdamConvNet/dense_5/bias/Adam_1ConvNet/dense_5/kernelConvNet/dense_5/kernel/AdamConvNet/dense_5/kernel/Adam_1ConvNet/dense_6/biasConvNet/dense_6/bias/AdamConvNet/dense_6/bias/Adam_1ConvNet/dense_6/kernelConvNet/dense_6/kernel/AdamConvNet/dense_6/kernel/Adam_1ConvNet/dense_7/biasConvNet/dense_7/bias/AdamConvNet/dense_7/bias/Adam_1ConvNet/dense_7/kernelConvNet/dense_7/kernel/AdamConvNet/dense_7/kernel/Adam_1Variablebeta1_powerbeta2_power"/device:CPU:0*G
dtypes=
;29
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
Ź
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
č
save/RestoreV2/tensor_namesConst"/device:CPU:0*
value˙Bü9BConvNet/cnn_logits_out/biasB ConvNet/cnn_logits_out/bias/AdamB"ConvNet/cnn_logits_out/bias/Adam_1BConvNet/cnn_logits_out/kernelB"ConvNet/cnn_logits_out/kernel/AdamB$ConvNet/cnn_logits_out/kernel/Adam_1BConvNet/dense/biasBConvNet/dense/bias/AdamBConvNet/dense/bias/Adam_1BConvNet/dense/kernelBConvNet/dense/kernel/AdamBConvNet/dense/kernel/Adam_1BConvNet/dense_1/biasBConvNet/dense_1/bias/AdamBConvNet/dense_1/bias/Adam_1BConvNet/dense_1/kernelBConvNet/dense_1/kernel/AdamBConvNet/dense_1/kernel/Adam_1BConvNet/dense_2/biasBConvNet/dense_2/bias/AdamBConvNet/dense_2/bias/Adam_1BConvNet/dense_2/kernelBConvNet/dense_2/kernel/AdamBConvNet/dense_2/kernel/Adam_1BConvNet/dense_3/biasBConvNet/dense_3/bias/AdamBConvNet/dense_3/bias/Adam_1BConvNet/dense_3/kernelBConvNet/dense_3/kernel/AdamBConvNet/dense_3/kernel/Adam_1BConvNet/dense_4/biasBConvNet/dense_4/bias/AdamBConvNet/dense_4/bias/Adam_1BConvNet/dense_4/kernelBConvNet/dense_4/kernel/AdamBConvNet/dense_4/kernel/Adam_1BConvNet/dense_5/biasBConvNet/dense_5/bias/AdamBConvNet/dense_5/bias/Adam_1BConvNet/dense_5/kernelBConvNet/dense_5/kernel/AdamBConvNet/dense_5/kernel/Adam_1BConvNet/dense_6/biasBConvNet/dense_6/bias/AdamBConvNet/dense_6/bias/Adam_1BConvNet/dense_6/kernelBConvNet/dense_6/kernel/AdamBConvNet/dense_6/kernel/Adam_1BConvNet/dense_7/biasBConvNet/dense_7/bias/AdamBConvNet/dense_7/bias/Adam_1BConvNet/dense_7/kernelBConvNet/dense_7/kernel/AdamBConvNet/dense_7/kernel/Adam_1BVariableBbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:9
č
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:9
ş
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*G
dtypes=
;29*ú
_output_shapesç
ä:::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Ŕ
save/AssignAssignConvNet/cnn_logits_out/biassave/RestoreV2*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:*
use_locking(
É
save/Assign_1Assign ConvNet/cnn_logits_out/bias/Adamsave/RestoreV2:1*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:
Ë
save/Assign_2Assign"ConvNet/cnn_logits_out/bias/Adam_1save/RestoreV2:2*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
:
Í
save/Assign_3AssignConvNet/cnn_logits_out/kernelsave/RestoreV2:3*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel
Ň
save/Assign_4Assign"ConvNet/cnn_logits_out/kernel/Adamsave/RestoreV2:4*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
Ô
save/Assign_5Assign$ConvNet/cnn_logits_out/kernel/Adam_1save/RestoreV2:5*
use_locking(*
T0*0
_class&
$"loc:@ConvNet/cnn_logits_out/kernel*
validate_shape(*
_output_shapes
:	
ł
save/Assign_6AssignConvNet/dense/biassave/RestoreV2:6*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:
¸
save/Assign_7AssignConvNet/dense/bias/Adamsave/RestoreV2:7*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ş
save/Assign_8AssignConvNet/dense/bias/Adam_1save/RestoreV2:8*
use_locking(*
T0*%
_class
loc:@ConvNet/dense/bias*
validate_shape(*
_output_shapes	
:
ź
save/Assign_9AssignConvNet/dense/kernelsave/RestoreV2:9* 
_output_shapes
:
*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(
Ă
save/Assign_10AssignConvNet/dense/kernel/Adamsave/RestoreV2:10*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:

Ĺ
save/Assign_11AssignConvNet/dense/kernel/Adam_1save/RestoreV2:11*
use_locking(*
T0*'
_class
loc:@ConvNet/dense/kernel*
validate_shape(* 
_output_shapes
:

š
save/Assign_12AssignConvNet/dense_1/biassave/RestoreV2:12*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_13AssignConvNet/dense_1/bias/Adamsave/RestoreV2:13*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:
Ŕ
save/Assign_14AssignConvNet/dense_1/bias/Adam_1save/RestoreV2:14*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_1/bias*
validate_shape(*
_output_shapes	
:
Â
save/Assign_15AssignConvNet/dense_1/kernelsave/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(* 
_output_shapes
:

Ç
save/Assign_16AssignConvNet/dense_1/kernel/Adamsave/RestoreV2:16*)
_class
loc:@ConvNet/dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
É
save/Assign_17AssignConvNet/dense_1/kernel/Adam_1save/RestoreV2:17*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_1/kernel
š
save/Assign_18AssignConvNet/dense_2/biassave/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_2/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_19AssignConvNet/dense_2/bias/Adamsave/RestoreV2:19*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_2/bias
Ŕ
save/Assign_20AssignConvNet/dense_2/bias/Adam_1save/RestoreV2:20*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_2/bias*
validate_shape(*
_output_shapes	
:
Â
save/Assign_21AssignConvNet/dense_2/kernelsave/RestoreV2:21* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_2/kernel*
validate_shape(
Ç
save/Assign_22AssignConvNet/dense_2/kernel/Adamsave/RestoreV2:22*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_2/kernel*
validate_shape(* 
_output_shapes
:

É
save/Assign_23AssignConvNet/dense_2/kernel/Adam_1save/RestoreV2:23*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_2/kernel*
validate_shape(* 
_output_shapes
:

š
save/Assign_24AssignConvNet/dense_3/biassave/RestoreV2:24*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_3/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_25AssignConvNet/dense_3/bias/Adamsave/RestoreV2:25*
T0*'
_class
loc:@ConvNet/dense_3/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ŕ
save/Assign_26AssignConvNet/dense_3/bias/Adam_1save/RestoreV2:26*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_3/bias
Â
save/Assign_27AssignConvNet/dense_3/kernelsave/RestoreV2:27*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_3/kernel*
validate_shape(* 
_output_shapes
:

Ç
save/Assign_28AssignConvNet/dense_3/kernel/Adamsave/RestoreV2:28*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_3/kernel
É
save/Assign_29AssignConvNet/dense_3/kernel/Adam_1save/RestoreV2:29*)
_class
loc:@ConvNet/dense_3/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
š
save/Assign_30AssignConvNet/dense_4/biassave/RestoreV2:30*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_4/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_31AssignConvNet/dense_4/bias/Adamsave/RestoreV2:31*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_4/bias*
validate_shape(*
_output_shapes	
:
Ŕ
save/Assign_32AssignConvNet/dense_4/bias/Adam_1save/RestoreV2:32*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_4/bias*
validate_shape(*
_output_shapes	
:
Â
save/Assign_33AssignConvNet/dense_4/kernelsave/RestoreV2:33*
T0*)
_class
loc:@ConvNet/dense_4/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ç
save/Assign_34AssignConvNet/dense_4/kernel/Adamsave/RestoreV2:34* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_4/kernel*
validate_shape(
É
save/Assign_35AssignConvNet/dense_4/kernel/Adam_1save/RestoreV2:35*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_4/kernel
š
save/Assign_36AssignConvNet/dense_5/biassave/RestoreV2:36*
T0*'
_class
loc:@ConvNet/dense_5/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ž
save/Assign_37AssignConvNet/dense_5/bias/Adamsave/RestoreV2:37*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_5/bias*
validate_shape(*
_output_shapes	
:
Ŕ
save/Assign_38AssignConvNet/dense_5/bias/Adam_1save/RestoreV2:38*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_5/bias
Â
save/Assign_39AssignConvNet/dense_5/kernelsave/RestoreV2:39*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_5/kernel*
validate_shape(* 
_output_shapes
:

Ç
save/Assign_40AssignConvNet/dense_5/kernel/Adamsave/RestoreV2:40*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_5/kernel
É
save/Assign_41AssignConvNet/dense_5/kernel/Adam_1save/RestoreV2:41*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_5/kernel*
validate_shape(* 
_output_shapes
:

š
save/Assign_42AssignConvNet/dense_6/biassave/RestoreV2:42*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_6/bias*
validate_shape(
ž
save/Assign_43AssignConvNet/dense_6/bias/Adamsave/RestoreV2:43*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_6/bias*
validate_shape(*
_output_shapes	
:
Ŕ
save/Assign_44AssignConvNet/dense_6/bias/Adam_1save/RestoreV2:44*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_6/bias*
validate_shape(*
_output_shapes	
:
Â
save/Assign_45AssignConvNet/dense_6/kernelsave/RestoreV2:45*)
_class
loc:@ConvNet/dense_6/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ç
save/Assign_46AssignConvNet/dense_6/kernel/Adamsave/RestoreV2:46*
T0*)
_class
loc:@ConvNet/dense_6/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
É
save/Assign_47AssignConvNet/dense_6/kernel/Adam_1save/RestoreV2:47*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_6/kernel*
validate_shape(* 
_output_shapes
:

š
save/Assign_48AssignConvNet/dense_7/biassave/RestoreV2:48*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_7/bias
ž
save/Assign_49AssignConvNet/dense_7/bias/Adamsave/RestoreV2:49*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_7/bias
Ŕ
save/Assign_50AssignConvNet/dense_7/bias/Adam_1save/RestoreV2:50*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@ConvNet/dense_7/bias*
validate_shape(
Â
save/Assign_51AssignConvNet/dense_7/kernelsave/RestoreV2:51* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_7/kernel*
validate_shape(
Ç
save/Assign_52AssignConvNet/dense_7/kernel/Adamsave/RestoreV2:52*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_7/kernel*
validate_shape(* 
_output_shapes
:

É
save/Assign_53AssignConvNet/dense_7/kernel/Adam_1save/RestoreV2:53*
use_locking(*
T0*)
_class
loc:@ConvNet/dense_7/kernel*
validate_shape(* 
_output_shapes
:

¤
save/Assign_54AssignVariablesave/RestoreV2:54*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(*
_output_shapes

:
˛
save/Assign_55Assignbeta1_powersave/RestoreV2:55*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
˛
save/Assign_56Assignbeta2_powersave/RestoreV2:56*
use_locking(*
T0*.
_class$
" loc:@ConvNet/cnn_logits_out/bias*
validate_shape(*
_output_shapes
: 
×
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shardÇ


 Dataset_flat_map_read_one_file_4
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
dtype0	*
valueB		 RY
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
buffer_sizeConst*
dtype0	*
valueB		 RY
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0


 Dataset_map_transform_to_orig_10
arg0
reshape
	reshape_1	2DWrapper for passing nested structures to and from tf.data functions.¸
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0*

dense_keys
 *
sparse_types
2	*
dense_shapes
 *
sparse_keys
XY*
Tdense
 *

num_sparseH
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
	reshape_1	2DWrapper for passing nested structures to and from tf.data functions.¸
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0*
sparse_types
2	*
dense_shapes
 *
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
Reshape_1/shapeConst*
valueB:*
dtype0^
	Reshape_1ReshapeSparseToDense_1:dense:0Reshape_1/shape:output:0*
T0	*
Tshape0"
	reshape_1Reshape_1:output:0"
reshapeReshape:output:0"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
train_op

Adam"ßA
	variablesŃAÎA

ConvNet/dense/kernel:0ConvNet/dense/kernel/AssignConvNet/dense/kernel/read:021ConvNet/dense/kernel/Initializer/random_uniform:08
v
ConvNet/dense/bias:0ConvNet/dense/bias/AssignConvNet/dense/bias/read:02&ConvNet/dense/bias/Initializer/zeros:08

ConvNet/dense_1/kernel:0ConvNet/dense_1/kernel/AssignConvNet/dense_1/kernel/read:023ConvNet/dense_1/kernel/Initializer/random_uniform:08
~
ConvNet/dense_1/bias:0ConvNet/dense_1/bias/AssignConvNet/dense_1/bias/read:02(ConvNet/dense_1/bias/Initializer/zeros:08

ConvNet/dense_2/kernel:0ConvNet/dense_2/kernel/AssignConvNet/dense_2/kernel/read:023ConvNet/dense_2/kernel/Initializer/random_uniform:08
~
ConvNet/dense_2/bias:0ConvNet/dense_2/bias/AssignConvNet/dense_2/bias/read:02(ConvNet/dense_2/bias/Initializer/zeros:08

ConvNet/dense_3/kernel:0ConvNet/dense_3/kernel/AssignConvNet/dense_3/kernel/read:023ConvNet/dense_3/kernel/Initializer/random_uniform:08
~
ConvNet/dense_3/bias:0ConvNet/dense_3/bias/AssignConvNet/dense_3/bias/read:02(ConvNet/dense_3/bias/Initializer/zeros:08

ConvNet/dense_4/kernel:0ConvNet/dense_4/kernel/AssignConvNet/dense_4/kernel/read:023ConvNet/dense_4/kernel/Initializer/random_uniform:08
~
ConvNet/dense_4/bias:0ConvNet/dense_4/bias/AssignConvNet/dense_4/bias/read:02(ConvNet/dense_4/bias/Initializer/zeros:08

ConvNet/dense_5/kernel:0ConvNet/dense_5/kernel/AssignConvNet/dense_5/kernel/read:023ConvNet/dense_5/kernel/Initializer/random_uniform:08
~
ConvNet/dense_5/bias:0ConvNet/dense_5/bias/AssignConvNet/dense_5/bias/read:02(ConvNet/dense_5/bias/Initializer/zeros:08

ConvNet/dense_6/kernel:0ConvNet/dense_6/kernel/AssignConvNet/dense_6/kernel/read:023ConvNet/dense_6/kernel/Initializer/random_uniform:08
~
ConvNet/dense_6/bias:0ConvNet/dense_6/bias/AssignConvNet/dense_6/bias/read:02(ConvNet/dense_6/bias/Initializer/zeros:08

ConvNet/dense_7/kernel:0ConvNet/dense_7/kernel/AssignConvNet/dense_7/kernel/read:023ConvNet/dense_7/kernel/Initializer/random_uniform:08
~
ConvNet/dense_7/bias:0ConvNet/dense_7/bias/AssignConvNet/dense_7/bias/read:02(ConvNet/dense_7/bias/Initializer/zeros:08
Ť
ConvNet/cnn_logits_out/kernel:0$ConvNet/cnn_logits_out/kernel/Assign$ConvNet/cnn_logits_out/kernel/read:02:ConvNet/cnn_logits_out/kernel/Initializer/random_uniform:08

ConvNet/cnn_logits_out/bias:0"ConvNet/cnn_logits_out/bias/Assign"ConvNet/cnn_logits_out/bias/read:02/ConvNet/cnn_logits_out/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
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

ConvNet/dense_2/kernel/Adam:0"ConvNet/dense_2/kernel/Adam/Assign"ConvNet/dense_2/kernel/Adam/read:02/ConvNet/dense_2/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_2/kernel/Adam_1:0$ConvNet/dense_2/kernel/Adam_1/Assign$ConvNet/dense_2/kernel/Adam_1/read:021ConvNet/dense_2/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_2/bias/Adam:0 ConvNet/dense_2/bias/Adam/Assign ConvNet/dense_2/bias/Adam/read:02-ConvNet/dense_2/bias/Adam/Initializer/zeros:0

ConvNet/dense_2/bias/Adam_1:0"ConvNet/dense_2/bias/Adam_1/Assign"ConvNet/dense_2/bias/Adam_1/read:02/ConvNet/dense_2/bias/Adam_1/Initializer/zeros:0

ConvNet/dense_3/kernel/Adam:0"ConvNet/dense_3/kernel/Adam/Assign"ConvNet/dense_3/kernel/Adam/read:02/ConvNet/dense_3/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_3/kernel/Adam_1:0$ConvNet/dense_3/kernel/Adam_1/Assign$ConvNet/dense_3/kernel/Adam_1/read:021ConvNet/dense_3/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_3/bias/Adam:0 ConvNet/dense_3/bias/Adam/Assign ConvNet/dense_3/bias/Adam/read:02-ConvNet/dense_3/bias/Adam/Initializer/zeros:0

ConvNet/dense_3/bias/Adam_1:0"ConvNet/dense_3/bias/Adam_1/Assign"ConvNet/dense_3/bias/Adam_1/read:02/ConvNet/dense_3/bias/Adam_1/Initializer/zeros:0

ConvNet/dense_4/kernel/Adam:0"ConvNet/dense_4/kernel/Adam/Assign"ConvNet/dense_4/kernel/Adam/read:02/ConvNet/dense_4/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_4/kernel/Adam_1:0$ConvNet/dense_4/kernel/Adam_1/Assign$ConvNet/dense_4/kernel/Adam_1/read:021ConvNet/dense_4/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_4/bias/Adam:0 ConvNet/dense_4/bias/Adam/Assign ConvNet/dense_4/bias/Adam/read:02-ConvNet/dense_4/bias/Adam/Initializer/zeros:0

ConvNet/dense_4/bias/Adam_1:0"ConvNet/dense_4/bias/Adam_1/Assign"ConvNet/dense_4/bias/Adam_1/read:02/ConvNet/dense_4/bias/Adam_1/Initializer/zeros:0

ConvNet/dense_5/kernel/Adam:0"ConvNet/dense_5/kernel/Adam/Assign"ConvNet/dense_5/kernel/Adam/read:02/ConvNet/dense_5/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_5/kernel/Adam_1:0$ConvNet/dense_5/kernel/Adam_1/Assign$ConvNet/dense_5/kernel/Adam_1/read:021ConvNet/dense_5/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_5/bias/Adam:0 ConvNet/dense_5/bias/Adam/Assign ConvNet/dense_5/bias/Adam/read:02-ConvNet/dense_5/bias/Adam/Initializer/zeros:0

ConvNet/dense_5/bias/Adam_1:0"ConvNet/dense_5/bias/Adam_1/Assign"ConvNet/dense_5/bias/Adam_1/read:02/ConvNet/dense_5/bias/Adam_1/Initializer/zeros:0

ConvNet/dense_6/kernel/Adam:0"ConvNet/dense_6/kernel/Adam/Assign"ConvNet/dense_6/kernel/Adam/read:02/ConvNet/dense_6/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_6/kernel/Adam_1:0$ConvNet/dense_6/kernel/Adam_1/Assign$ConvNet/dense_6/kernel/Adam_1/read:021ConvNet/dense_6/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_6/bias/Adam:0 ConvNet/dense_6/bias/Adam/Assign ConvNet/dense_6/bias/Adam/read:02-ConvNet/dense_6/bias/Adam/Initializer/zeros:0

ConvNet/dense_6/bias/Adam_1:0"ConvNet/dense_6/bias/Adam_1/Assign"ConvNet/dense_6/bias/Adam_1/read:02/ConvNet/dense_6/bias/Adam_1/Initializer/zeros:0

ConvNet/dense_7/kernel/Adam:0"ConvNet/dense_7/kernel/Adam/Assign"ConvNet/dense_7/kernel/Adam/read:02/ConvNet/dense_7/kernel/Adam/Initializer/zeros:0
 
ConvNet/dense_7/kernel/Adam_1:0$ConvNet/dense_7/kernel/Adam_1/Assign$ConvNet/dense_7/kernel/Adam_1/read:021ConvNet/dense_7/kernel/Adam_1/Initializer/zeros:0

ConvNet/dense_7/bias/Adam:0 ConvNet/dense_7/bias/Adam/Assign ConvNet/dense_7/bias/Adam/read:02-ConvNet/dense_7/bias/Adam/Initializer/zeros:0

ConvNet/dense_7/bias/Adam_1:0"ConvNet/dense_7/bias/Adam_1/Assign"ConvNet/dense_7/bias/Adam_1/read:02/ConvNet/dense_7/bias/Adam_1/Initializer/zeros:0
´
$ConvNet/cnn_logits_out/kernel/Adam:0)ConvNet/cnn_logits_out/kernel/Adam/Assign)ConvNet/cnn_logits_out/kernel/Adam/read:026ConvNet/cnn_logits_out/kernel/Adam/Initializer/zeros:0
ź
&ConvNet/cnn_logits_out/kernel/Adam_1:0+ConvNet/cnn_logits_out/kernel/Adam_1/Assign+ConvNet/cnn_logits_out/kernel/Adam_1/read:028ConvNet/cnn_logits_out/kernel/Adam_1/Initializer/zeros:0
Ź
"ConvNet/cnn_logits_out/bias/Adam:0'ConvNet/cnn_logits_out/bias/Adam/Assign'ConvNet/cnn_logits_out/bias/Adam/read:024ConvNet/cnn_logits_out/bias/Adam/Initializer/zeros:0
´
$ConvNet/cnn_logits_out/bias/Adam_1:0)ConvNet/cnn_logits_out/bias/Adam_1/Assign)ConvNet/cnn_logits_out/bias/Adam_1/read:026ConvNet/cnn_logits_out/bias/Adam_1/Initializer/zeros:0
9

Variable:0Variable/AssignVariable/read:02zeros:08"-
	iterators 

IteratorV2:0
IteratorV2_1:0"Ą
trainable_variables

ConvNet/dense/kernel:0ConvNet/dense/kernel/AssignConvNet/dense/kernel/read:021ConvNet/dense/kernel/Initializer/random_uniform:08
v
ConvNet/dense/bias:0ConvNet/dense/bias/AssignConvNet/dense/bias/read:02&ConvNet/dense/bias/Initializer/zeros:08

ConvNet/dense_1/kernel:0ConvNet/dense_1/kernel/AssignConvNet/dense_1/kernel/read:023ConvNet/dense_1/kernel/Initializer/random_uniform:08
~
ConvNet/dense_1/bias:0ConvNet/dense_1/bias/AssignConvNet/dense_1/bias/read:02(ConvNet/dense_1/bias/Initializer/zeros:08

ConvNet/dense_2/kernel:0ConvNet/dense_2/kernel/AssignConvNet/dense_2/kernel/read:023ConvNet/dense_2/kernel/Initializer/random_uniform:08
~
ConvNet/dense_2/bias:0ConvNet/dense_2/bias/AssignConvNet/dense_2/bias/read:02(ConvNet/dense_2/bias/Initializer/zeros:08

ConvNet/dense_3/kernel:0ConvNet/dense_3/kernel/AssignConvNet/dense_3/kernel/read:023ConvNet/dense_3/kernel/Initializer/random_uniform:08
~
ConvNet/dense_3/bias:0ConvNet/dense_3/bias/AssignConvNet/dense_3/bias/read:02(ConvNet/dense_3/bias/Initializer/zeros:08

ConvNet/dense_4/kernel:0ConvNet/dense_4/kernel/AssignConvNet/dense_4/kernel/read:023ConvNet/dense_4/kernel/Initializer/random_uniform:08
~
ConvNet/dense_4/bias:0ConvNet/dense_4/bias/AssignConvNet/dense_4/bias/read:02(ConvNet/dense_4/bias/Initializer/zeros:08

ConvNet/dense_5/kernel:0ConvNet/dense_5/kernel/AssignConvNet/dense_5/kernel/read:023ConvNet/dense_5/kernel/Initializer/random_uniform:08
~
ConvNet/dense_5/bias:0ConvNet/dense_5/bias/AssignConvNet/dense_5/bias/read:02(ConvNet/dense_5/bias/Initializer/zeros:08

ConvNet/dense_6/kernel:0ConvNet/dense_6/kernel/AssignConvNet/dense_6/kernel/read:023ConvNet/dense_6/kernel/Initializer/random_uniform:08
~
ConvNet/dense_6/bias:0ConvNet/dense_6/bias/AssignConvNet/dense_6/bias/read:02(ConvNet/dense_6/bias/Initializer/zeros:08

ConvNet/dense_7/kernel:0ConvNet/dense_7/kernel/AssignConvNet/dense_7/kernel/read:023ConvNet/dense_7/kernel/Initializer/random_uniform:08
~
ConvNet/dense_7/bias:0ConvNet/dense_7/bias/AssignConvNet/dense_7/bias/read:02(ConvNet/dense_7/bias/Initializer/zeros:08
Ť
ConvNet/cnn_logits_out/kernel:0$ConvNet/cnn_logits_out/kernel/Assign$ConvNet/cnn_logits_out/kernel/read:02:ConvNet/cnn_logits_out/kernel/Initializer/random_uniform:08

ConvNet/cnn_logits_out/bias:0"ConvNet/cnn_logits_out/bias/Assign"ConvNet/cnn_logits_out/bias/read:02/ConvNet/cnn_logits_out/bias/Initializer/zeros:08
9

Variable:0Variable/AssignVariable/read:02zeros:08