шн
р
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
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
Ў
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
+
IsNan
x"T
y
"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	


LogicalNot
x

y

>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
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
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58Ќ
ж
ConstConst*
_output_shapes
:a*
dtype0*
valueBa"      ?   @  @@  @   @  Р@  р@   A  A   A  0A  @A  PA  `A  pA  A  A  A  A   A  ЈA  АA  ИA  РA  ШA  аA  иA  рA  шA  №A  јA   B  B  B  B  B  B  B  B   B  $B  (B  ,B  0B  4B  8B  <B  @B  DB  HB  LB  PB  TB  XB  \B  `B  dB  hB  lB  pB  tB  xB  |B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B   B  ЂB  ЄB  ІB  ЈB  ЊB  ЌB  ЎB  АB  ВB  ДB  ЖB  ИB  КB  МB  ОB  РB

serving_default_inputsPlaceholder*,
_output_shapes
:џџџџџџџџџ*
dtype0*!
shape:џџџџџџџџџ
Ћ
PartitionedCallPartitionedCallserving_default_inputsConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_97636

NoOpNoOp
Я
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueўBћ Bє
е
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
flatten* 
* 
* 
* 
Ў
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

	capture_0* 

serving_default* 
* 
* 
* 

non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

"trace_0* 

#trace_0* 

$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
* 

0
1* 
* 
* 
* 

	capture_0* 

	capture_0* 

	capture_0* 

	capture_0* 
* 

	capture_0* 
* 
	
0* 
* 
* 
* 

	capture_0* 

	capture_0* 
* 
* 
* 

*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_98076

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_98086Ўч
а
z
B__inference_model_1_layer_call_and_return_conditional_losses_97627

inputs
trans_preprocess_97623
identityх
 trans_preprocess/PartitionedCallPartitionedCallinputstrans_preprocess_97623*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_97567v
IdentityIdentity)trans_preprocess/PartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
ѕ
V
0__inference_trans_preprocess_layer_call_fn_97643
pos
unknown
identityТ
PartitionedCallPartitionedCallposunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_97567e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:Q M
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepos: 

_output_shapes
:a
а
z
B__inference_model_1_layer_call_and_return_conditional_losses_97572

inputs
trans_preprocess_97568
identityх
 trans_preprocess/PartitionedCallPartitionedCallinputstrans_preprocess_97568*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_97567v
IdentityIdentity)trans_preprocess/PartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
І
q
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_97567
pos
unknown
identity8
ShapeShapepos*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
add/yConst*
_output_shapes
: *
dtype0*
value	B :U
addAddV2strided_slice:output:0add/y:output:0*
T0*
_output_shapes
: G
ConstConst*
_output_shapes
: *
dtype0*
value	B :aR
Const_1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџJ
Shape_1Const*
_output_shapes
: *
dtype0*
valueB J
Shape_2Const*
_output_shapes
: *
dtype0*
valueB \
BroadcastArgsBroadcastArgsShape_1:output:0Shape_2:output:0*
_output_shapes
: J
Shape_3Const*
_output_shapes
: *
dtype0*
valueB `
BroadcastArgs_1BroadcastArgsBroadcastArgs:r0:0Shape_3:output:0*
_output_shapes
: a
BroadcastToBroadcastToConst:output:0BroadcastArgs_1:r0:0*
T0*
_output_shapes
: e
BroadcastTo_1BroadcastToConst_1:output:0BroadcastArgs_1:r0:0*
T0*
_output_shapes
: \
BroadcastTo_2BroadcastToadd:z:0BroadcastArgs_1:r0:0*
T0*
_output_shapes
: o
clip_by_value/MinimumMinimumBroadcastTo:output:0BroadcastTo_2:output:0*
T0*
_output_shapes
: l
clip_by_valueMaximumclip_by_value/Minimum:z:0BroadcastTo_1:output:0*
T0*
_output_shapes
: G
sub/yConst*
_output_shapes
: *
dtype0*
value	B :N
subSubclip_by_value:z:0sub/y:output:0*
T0*
_output_shapes
: I
Const_2Const*
_output_shapes
: *
dtype0*
value	B : I
Const_3Const*
_output_shapes
: *
dtype0*
value	B :]
strided_slice_1/stackPackConst_2:output:0*
N*
T0*
_output_shapes
:V
strided_slice_1/stack_1Packsub:z:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stack_2PackConst_3:output:0*
N*
T0*
_output_shapes
:й
strided_slice_1StridedSliceunknownstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_maskT
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: \
mulMulstrided_slice_1:output:0Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :R
sub_1Subclip_by_value:z:0sub_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
truedivRealDivmul:z:0
Cast_1:y:0*
T0*#
_output_shapes
:џџџџџџџџџX
Cast_2Casttruediv:z:0*

DstT0*

SrcT0*#
_output_shapes
:џџџџџџџџџ:
Shape_4Shapepos*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_4:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Y
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0*
_output_shapes
: g
clip_by_value_1/MinimumMinimum
Cast_2:y:0	sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџS
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
GatherV2GatherV2posclip_by_value_1:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:џџџџџџџџџ
Const_4Const*
_output_shapes
:(*
dtype0	*и
valueЮBЫ	("Р                                                     #      8      7      6            4      =            >      D      :            A      w      %       '       (       Й       =       R       Q       P       П       N       W       В       X       _       T       Е       [              X
Shape_5ShapeGatherV2:output:0*
T0*
_output_shapes
:*
out_type0	_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_5:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: d
clip_by_value_2/MinimumMinimumConst_4:output:0	sub_3:z:0*
T0	*
_output_shapes
:(S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
:(Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_1GatherV2GatherV2:output:0clip_by_value_2:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ(f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    д  h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    щ  h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ћ
strided_slice_4StridedSliceGatherV2:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskf
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    
  h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ћ
strided_slice_5StridedSliceGatherV2:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskj
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          l
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         §
strided_slice_6StridedSliceGatherV2_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @n
mul_1Mulmul_1/x:output:0strided_slice_6:output:0*
T0*+
_output_shapes
:џџџџџџџџџf
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ѕ
strided_slice_7StridedSlicestrided_slice_5:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
ellipsis_maskg
sub_4Sub	mul_1:z:0strided_slice_7:output:0*
T0*+
_output_shapes
:џџџџџџџџџf
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_8StridedSlicestrided_slice_5:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
end_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
concatConcatV2	sub_4:z:0strided_slice_8:output:0concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ^
IsNanIsNanstrided_slice_4:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
Cast_3Cast	IsNan:y:0*

DstT0	*

SrcT0
*+
_output_shapes
:џџџџџџџџџ\
Const_5Const*
_output_shapes
:*
dtype0*!
valueB"          I
SumSum
Cast_3:y:0Const_5:output:0*
T0	*
_output_shapes
: W
IsNan_1IsNanconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
Cast_4CastIsNan_1:y:0*

DstT0	*

SrcT0
*+
_output_shapes
:џџџџџџџџџ\
Const_6Const*
_output_shapes
:*
dtype0*!
valueB"          K
Sum_1Sum
Cast_4:y:0Const_6:output:0*
T0	*
_output_shapes
: K
LessLessSum:output:0Sum_1:output:0*
T0	*
_output_shapes
: 
SelectV2SelectV2Less:z:0strided_slice_4:output:0concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџц
Const_7Const*
_output_shapes	
:в*
dtype0	*Љ
valueB	в"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    	       	       	       	       	       	       	       	       	       	       	       
       
       
       
       
       
       
       
       
       
                                                                                                                                                                                                                                                                                                                                  X
Shape_6ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	_
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_9StridedSliceShape_6:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_5Substrided_slice_9:output:0sub_5/y:output:0*
T0	*
_output_shapes
: e
clip_by_value_3/MinimumMinimumConst_7:output:0	sub_5:z:0*
T0	*
_output_shapes	
:вS
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes	
:вQ
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B :В

GatherV2_2GatherV2SelectV2:output:0clip_by_value_3:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџвg
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_10StridedSliceGatherV2_2:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџв*

begin_mask*
ellipsis_maskц
Const_8Const*
_output_shapes	
:в*
dtype0	*Љ
valueB	в"                                                        	       
                                                                                                                              	       
                                                                                                                       	       
                                                                                                                	       
                                                                                                         	       
                                                                                                  	       
                                                                                           	       
                                                                                    	       
                                                                             	       
                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              X
Shape_7ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
strided_slice_11StridedSliceShape_7:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 RZ
sub_6Substrided_slice_11:output:0sub_6/y:output:0*
T0	*
_output_shapes
: e
clip_by_value_4/MinimumMinimumConst_8:output:0	sub_6:z:0*
T0	*
_output_shapes	
:вS
clip_by_value_4/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0	*
_output_shapes	
:вQ
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B :В

GatherV2_3GatherV2SelectV2:output:0clip_by_value_4:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџвg
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_12StridedSliceGatherV2_3:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџв*

begin_mask*
ellipsis_masky
sub_7Substrided_slice_10:output:0strided_slice_12:output:0*
T0*,
_output_shapes
:џџџџџџџџџв\
norm/mulMul	sub_7:z:0	sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџвm
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџв*
	keep_dims([
	norm/SqrtSqrtnorm/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџвy
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџв*
squeeze_dims

џџџџџџџџџЦ
Const_9Const*
_output_shapes	
:О*
dtype0	*
valueџBќ	О"№                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    	       	       	       	       	       	       	       	       	       	       
       
       
       
       
       
       
       
       
                                                                                                                                                                                                                                                                   Z
Shape_8ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
strided_slice_13StridedSliceShape_8:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 RZ
sub_8Substrided_slice_13:output:0sub_8/y:output:0*
T0	*
_output_shapes
: e
clip_by_value_5/MinimumMinimumConst_9:output:0	sub_8:z:0*
T0	*
_output_shapes	
:ОS
clip_by_value_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0	*
_output_shapes	
:ОQ
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B :Д

GatherV2_4GatherV2GatherV2_1:output:0clip_by_value_5:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџОg
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_14StridedSliceGatherV2_4:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџО*

begin_mask*
ellipsis_maskЧ
Const_10Const*
_output_shapes	
:О*
dtype0	*
valueџBќ	О"№                                                        	       
                                                                                                                       	       
                                                                                                                	       
                                                                                                         	       
                                                                                                  	       
                                                                                           	       
                                                                                    	       
                                                                             	       
                                                                      	       
                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                 Z
Shape_9ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
strided_slice_15StridedSliceShape_9:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 RZ
sub_9Substrided_slice_15:output:0sub_9/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_6/MinimumMinimumConst_10:output:0	sub_9:z:0*
T0	*
_output_shapes	
:ОS
clip_by_value_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0	*
_output_shapes	
:ОQ
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B :Д

GatherV2_5GatherV2GatherV2_1:output:0clip_by_value_6:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџОg
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_16StridedSliceGatherV2_5:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџО*

begin_mask*
ellipsis_maskz
sub_10Substrided_slice_14:output:0strided_slice_16:output:0*
T0*,
_output_shapes
:џџџџџџџџџО`

norm_1/mulMul
sub_10:z:0
sub_10:z:0*
T0*,
_output_shapes
:џџџџџџџџџОo
norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_1/SumSumnorm_1/mul:z:0%norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџО*
	keep_dims(_
norm_1/SqrtSqrtnorm_1/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџО}
norm_1/SqueezeSqueezenorm_1/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџО*
squeeze_dims

џџџџџџџџџЬ
Const_11Const*
_output_shapes
:*
dtype0	*
valueB	"x                                                    	       
                                                   Y
Shape_10ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_17StridedSliceShape_10:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_11Substrided_slice_17:output:0sub_11/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_7/MinimumMinimumConst_11:output:0
sub_11:z:0*
T0	*
_output_shapes
:S
clip_by_value_7/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_6GatherV2SelectV2:output:0clip_by_value_7:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџЬ
Const_12Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          	       
                                                        Y
Shape_11ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_18StridedSliceShape_11:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_12Substrided_slice_18:output:0sub_12/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_8/MinimumMinimumConst_12:output:0
sub_12:z:0*
T0	*
_output_shapes
:S
clip_by_value_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_7GatherV2SelectV2:output:0clip_by_value_8:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџm
sub_13SubGatherV2_6:output:0GatherV2_7:output:0*
T0*+
_output_shapes
:џџџџџџџџџЬ
Const_13Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          
                                                               Y
Shape_12ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_19StridedSliceShape_12:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_14Substrided_slice_19:output:0sub_14/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_9/MinimumMinimumConst_13:output:0
sub_14:z:0*
T0	*
_output_shapes
:S
clip_by_value_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_9Maximumclip_by_value_9/Minimum:z:0clip_by_value_9/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_8GatherV2SelectV2:output:0clip_by_value_9:z:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџЬ
Const_14Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          	       
                                                        Y
Shape_13ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_20StridedSliceShape_13:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_15/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_15Substrided_slice_20:output:0sub_15/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_10/MinimumMinimumConst_14:output:0
sub_15:z:0*
T0	*
_output_shapes
:T
clip_by_value_10/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_10Maximumclip_by_value_10/Minimum:z:0clip_by_value_10/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B :В

GatherV2_9GatherV2SelectV2:output:0clip_by_value_10:z:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџm
sub_16SubGatherV2_8:output:0GatherV2_9:output:0*
T0*+
_output_shapes
:џџџџџџџџџ_

norm_2/mulMul
sub_13:z:0
sub_13:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_2/SumSumnorm_2/mul:z:0%norm_2/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_2/SqrtSqrtnorm_2/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_2/SqueezeSqueezenorm_2/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџ_

norm_3/mulMul
sub_16:z:0
sub_16:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_3/SumSumnorm_3/mul:z:0%norm_3/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_3/SqrtSqrtnorm_3/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_3/SqueezeSqueezenorm_3/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџZ
mul_2Mul
sub_13:z:0
sub_16:z:0*
T0*+
_output_shapes
:џџџџџџџџџb
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџk
Sum_2Sum	mul_2:z:0 Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџo
	truediv_1RealDivSum_2:output:0norm_2/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
	truediv_2RealDivtruediv_1:z:0norm_3/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Const_15Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@        	                            "              %       [
Shape_14ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_21StridedSliceShape_14:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_17/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_17Substrided_slice_21:output:0sub_17/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_11/MinimumMinimumConst_15:output:0
sub_17:z:0*
T0	*
_output_shapes
:T
clip_by_value_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_11Maximumclip_by_value_11/Minimum:z:0clip_by_value_11/y:output:0*
T0	*
_output_shapes
:R
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_10GatherV2GatherV2_1:output:0clip_by_value_11:z:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
Const_16Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@	                             "              %              [
Shape_15ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_22StridedSliceShape_15:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_18/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_18Substrided_slice_22:output:0sub_18/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_12/MinimumMinimumConst_16:output:0
sub_18:z:0*
T0	*
_output_shapes
:T
clip_by_value_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_12Maximumclip_by_value_12/Minimum:z:0clip_by_value_12/y:output:0*
T0	*
_output_shapes
:R
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_11GatherV2GatherV2_1:output:0clip_by_value_12:z:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџo
sub_19SubGatherV2_10:output:0GatherV2_11:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
Const_17Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@                      	              %              "       [
Shape_16ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_23StridedSliceShape_16:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_20/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_20Substrided_slice_23:output:0sub_20/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_13/MinimumMinimumConst_17:output:0
sub_20:z:0*
T0	*
_output_shapes
:T
clip_by_value_13/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_13Maximumclip_by_value_13/Minimum:z:0clip_by_value_13/y:output:0*
T0	*
_output_shapes
:R
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_12GatherV2GatherV2_1:output:0clip_by_value_13:z:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
Const_18Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@	                             "              %              [
Shape_17ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_24StridedSliceShape_17:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_21/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_21Substrided_slice_24:output:0sub_21/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_14/MinimumMinimumConst_18:output:0
sub_21:z:0*
T0	*
_output_shapes
:T
clip_by_value_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_14Maximumclip_by_value_14/Minimum:z:0clip_by_value_14/y:output:0*
T0	*
_output_shapes
:R
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_13GatherV2GatherV2_1:output:0clip_by_value_14:z:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџo
sub_22SubGatherV2_12:output:0GatherV2_13:output:0*
T0*+
_output_shapes
:џџџџџџџџџ_

norm_4/mulMul
sub_19:z:0
sub_19:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_4/SumSumnorm_4/mul:z:0%norm_4/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_4/SqrtSqrtnorm_4/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_4/SqueezeSqueezenorm_4/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџ_

norm_5/mulMul
sub_22:z:0
sub_22:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_5/SumSumnorm_5/mul:z:0%norm_5/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_5/SqrtSqrtnorm_5/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_5/SqueezeSqueezenorm_5/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџZ
mul_3Mul
sub_19:z:0
sub_22:z:0*
T0*+
_output_shapes
:џџџџџџџџџb
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџk
Sum_3Sum	mul_3:z:0 Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџo
	truediv_3RealDivSum_3:output:0norm_4/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
	truediv_4RealDivtruediv_3:z:0norm_5/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2GatherV2_1:output:0SelectV2:output:0concat_1/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџk
ReshapeReshapeconcat_1:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџP
IsNan_2IsNanReshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџJ

LogicalNot
LogicalNotIsNan_2:y:0*#
_output_shapes
:џџџџџџџџџR
boolean_mask/ShapeShapeReshape:output:0*
T0*
_output_shapes
:j
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:m
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: T
boolean_mask/Shape_1ShapeReshape:output:0*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskT
boolean_mask/Shape_2ShapeReshape:output:0*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskn
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:Z
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : х
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:}
boolean_mask/ReshapeReshapeReshape:output:0boolean_mask/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџo
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask/Reshape_1ReshapeLogicalNot:y:0%boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџe
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
\
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : е
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџR
Const_19Const*
_output_shapes
:*
dtype0*
valueB: `
MeanMeanboolean_mask/GatherV2:output:0Const_19:output:0*
T0*
_output_shapes
: R
Const_20Const*
_output_shapes
:*
dtype0*
valueB: w
Mean_1Meanboolean_mask/GatherV2:output:0Const_20:output:0*
T0*
_output_shapes
:*
	keep_dims(l
sub_23Subboolean_mask/GatherV2:output:0Mean_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџJ
SquareSquare
sub_23:z:0*
T0*#
_output_shapes
:џџџџџџџџџR
Const_21Const*
_output_shapes
:*
dtype0*
valueB: L
Sum_4Sum
Square:y:0Const_21:output:0*
T0*
_output_shapes
: M
SizeSizeboolean_mask/GatherV2:output:0*
T0*
_output_shapes
: J
sub_24/yConst*
_output_shapes
: *
dtype0*
value	B :P
sub_24SubSize:output:0sub_24/y:output:0*
T0*
_output_shapes
: J
Cast_5Cast
sub_24:z:0*

DstT0*

SrcT0*
_output_shapes
: Q
	truediv_5RealDivSum_4:output:0
Cast_5:y:0*
T0*
_output_shapes
: <
SqrtSqrttruediv_5:z:0*
T0*
_output_shapes
: e
sub_25Subconcat_1:output:0Mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=`
	truediv_6RealDiv
sub_25:z:0Sqrt:y:0*
T0*+
_output_shapes
:џџџџџџџџџ=i
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_25StridedSlicetruediv_6:z:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*
end_maskh

zeros_like	ZerosLikestrided_slice_25:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=`
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
strided_slice_26StridedSlicetruediv_6:z:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*

begin_mask`
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_27StridedSlicetruediv_6:z:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*
end_masky
sub_26Substrided_slice_26:output:0strided_slice_27:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_2ConcatV2
sub_26:z:0zeros_like:y:0concat_2/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=L
NegNeg
sub_26:z:0*
T0*+
_output_shapes
:џџџџџџџџџ=O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2zeros_like:y:0Neg:y:0concat_3/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   x
flatten_1/ReshapeReshapetruediv_6:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗb
flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   
flatten_1/Reshape_1Reshapeconcat_2:output:0flatten_1/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗb
flatten_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   
flatten_1/Reshape_2Reshapeconcat_3:output:0flatten_1/Const_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗb
flatten_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"џџџџО   
flatten_1/Reshape_3Reshapenorm_1/Squeeze:output:0flatten_1/Const_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџОb
flatten_1/Const_4Const*
_output_shapes
:*
dtype0*
valueB"џџџџв   
flatten_1/Reshape_4Reshapenorm/Squeeze:output:0flatten_1/Const_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџвb
flatten_1/Const_5Const*
_output_shapes
:*
dtype0*
valueB"џџџџ   {
flatten_1/Reshape_5Reshapetruediv_4:z:0flatten_1/Const_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
flatten_1/Const_6Const*
_output_shapes
:*
dtype0*
valueB"џџџџ   {
flatten_1/Reshape_6Reshapetruediv_2:z:0flatten_1/Const_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
concat_4/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
concat_4ConcatV2flatten_1/Reshape:output:0flatten_1/Reshape_1:output:0flatten_1/Reshape_2:output:0flatten_1/Reshape_3:output:0flatten_1/Reshape_4:output:0flatten_1/Reshape_5:output:0flatten_1/Reshape_6:output:0concat_4/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬV
IsNan_3IsNanconcat_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬQ
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *    

SelectV2_1SelectV2IsNan_3:y:0SelectV2_1/t:output:0concat_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ`
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_28StridedSliceSelectV2_1:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџЬ*
new_axis_maskf
IdentityIdentitystrided_slice_28:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:Q M
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepos: 

_output_shapes
:a
І
q
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_98052
pos
unknown
identity8
ShapeShapepos*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
add/yConst*
_output_shapes
: *
dtype0*
value	B :U
addAddV2strided_slice:output:0add/y:output:0*
T0*
_output_shapes
: G
ConstConst*
_output_shapes
: *
dtype0*
value	B :aR
Const_1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџJ
Shape_1Const*
_output_shapes
: *
dtype0*
valueB J
Shape_2Const*
_output_shapes
: *
dtype0*
valueB \
BroadcastArgsBroadcastArgsShape_1:output:0Shape_2:output:0*
_output_shapes
: J
Shape_3Const*
_output_shapes
: *
dtype0*
valueB `
BroadcastArgs_1BroadcastArgsBroadcastArgs:r0:0Shape_3:output:0*
_output_shapes
: a
BroadcastToBroadcastToConst:output:0BroadcastArgs_1:r0:0*
T0*
_output_shapes
: e
BroadcastTo_1BroadcastToConst_1:output:0BroadcastArgs_1:r0:0*
T0*
_output_shapes
: \
BroadcastTo_2BroadcastToadd:z:0BroadcastArgs_1:r0:0*
T0*
_output_shapes
: o
clip_by_value/MinimumMinimumBroadcastTo:output:0BroadcastTo_2:output:0*
T0*
_output_shapes
: l
clip_by_valueMaximumclip_by_value/Minimum:z:0BroadcastTo_1:output:0*
T0*
_output_shapes
: G
sub/yConst*
_output_shapes
: *
dtype0*
value	B :N
subSubclip_by_value:z:0sub/y:output:0*
T0*
_output_shapes
: I
Const_2Const*
_output_shapes
: *
dtype0*
value	B : I
Const_3Const*
_output_shapes
: *
dtype0*
value	B :]
strided_slice_1/stackPackConst_2:output:0*
N*
T0*
_output_shapes
:V
strided_slice_1/stack_1Packsub:z:0*
N*
T0*
_output_shapes
:_
strided_slice_1/stack_2PackConst_3:output:0*
N*
T0*
_output_shapes
:й
strided_slice_1StridedSliceunknownstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_maskT
CastCaststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: \
mulMulstrided_slice_1:output:0Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџI
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :R
sub_1Subclip_by_value:z:0sub_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
truedivRealDivmul:z:0
Cast_1:y:0*
T0*#
_output_shapes
:џџџџџџџџџX
Cast_2Casttruediv:z:0*

DstT0*

SrcT0*#
_output_shapes
:џџџџџџџџџ:
Shape_4Shapepos*
T0*
_output_shapes
:_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_2StridedSliceShape_4:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Y
sub_2Substrided_slice_2:output:0sub_2/y:output:0*
T0*
_output_shapes
: g
clip_by_value_1/MinimumMinimum
Cast_2:y:0	sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџS
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
GatherV2GatherV2posclip_by_value_1:z:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:џџџџџџџџџ
Const_4Const*
_output_shapes
:(*
dtype0	*и
valueЮBЫ	("Р                                                     #      8      7      6            4      =            >      D      :            A      w      %       '       (       Й       =       R       Q       P       П       N       W       В       X       _       T       Е       [              X
Shape_5ShapeGatherV2:output:0*
T0*
_output_shapes
:*
out_type0	_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_3StridedSliceShape_5:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_3Substrided_slice_3:output:0sub_3/y:output:0*
T0	*
_output_shapes
: d
clip_by_value_2/MinimumMinimumConst_4:output:0	sub_3:z:0*
T0	*
_output_shapes
:(S
clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0	*
_output_shapes
:(Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_1GatherV2GatherV2:output:0clip_by_value_2:z:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ(f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    д  h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    щ  h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ћ
strided_slice_4StridedSliceGatherV2:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskf
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    
  h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ћ
strided_slice_5StridedSliceGatherV2:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskj
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            l
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          l
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         §
strided_slice_6StridedSliceGatherV2_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @n
mul_1Mulmul_1/x:output:0strided_slice_6:output:0*
T0*+
_output_shapes
:џџџџџџџџџf
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ѕ
strided_slice_7StridedSlicestrided_slice_5:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
ellipsis_maskg
sub_4Sub	mul_1:z:0strided_slice_7:output:0*
T0*+
_output_shapes
:џџџџџџџџџf
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_8StridedSlicestrided_slice_5:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
end_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
concatConcatV2	sub_4:z:0strided_slice_8:output:0concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ^
IsNanIsNanstrided_slice_4:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
Cast_3Cast	IsNan:y:0*

DstT0	*

SrcT0
*+
_output_shapes
:џџџџџџџџџ\
Const_5Const*
_output_shapes
:*
dtype0*!
valueB"          I
SumSum
Cast_3:y:0Const_5:output:0*
T0	*
_output_shapes
: W
IsNan_1IsNanconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`
Cast_4CastIsNan_1:y:0*

DstT0	*

SrcT0
*+
_output_shapes
:џџџџџџџџџ\
Const_6Const*
_output_shapes
:*
dtype0*!
valueB"          K
Sum_1Sum
Cast_4:y:0Const_6:output:0*
T0	*
_output_shapes
: K
LessLessSum:output:0Sum_1:output:0*
T0	*
_output_shapes
: 
SelectV2SelectV2Less:z:0strided_slice_4:output:0concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџц
Const_7Const*
_output_shapes	
:в*
dtype0	*Љ
valueB	в"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    	       	       	       	       	       	       	       	       	       	       	       
       
       
       
       
       
       
       
       
       
                                                                                                                                                                                                                                                                                                                                  X
Shape_6ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	_
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_9StridedSliceShape_6:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 RY
sub_5Substrided_slice_9:output:0sub_5/y:output:0*
T0	*
_output_shapes
: e
clip_by_value_3/MinimumMinimumConst_7:output:0	sub_5:z:0*
T0	*
_output_shapes	
:вS
clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_3Maximumclip_by_value_3/Minimum:z:0clip_by_value_3/y:output:0*
T0	*
_output_shapes	
:вQ
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B :В

GatherV2_2GatherV2SelectV2:output:0clip_by_value_3:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџвg
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_10StridedSliceGatherV2_2:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџв*

begin_mask*
ellipsis_maskц
Const_8Const*
_output_shapes	
:в*
dtype0	*Љ
valueB	в"                                                        	       
                                                                                                                              	       
                                                                                                                       	       
                                                                                                                	       
                                                                                                         	       
                                                                                                  	       
                                                                                           	       
                                                                                    	       
                                                                             	       
                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              X
Shape_7ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
strided_slice_11StridedSliceShape_7:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 RZ
sub_6Substrided_slice_11:output:0sub_6/y:output:0*
T0	*
_output_shapes
: e
clip_by_value_4/MinimumMinimumConst_8:output:0	sub_6:z:0*
T0	*
_output_shapes	
:вS
clip_by_value_4/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_4Maximumclip_by_value_4/Minimum:z:0clip_by_value_4/y:output:0*
T0	*
_output_shapes	
:вQ
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B :В

GatherV2_3GatherV2SelectV2:output:0clip_by_value_4:z:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџвg
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_12StridedSliceGatherV2_3:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџв*

begin_mask*
ellipsis_masky
sub_7Substrided_slice_10:output:0strided_slice_12:output:0*
T0*,
_output_shapes
:џџџџџџџџџв\
norm/mulMul	sub_7:z:0	sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџвm
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџв*
	keep_dims([
	norm/SqrtSqrtnorm/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџвy
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџв*
squeeze_dims

џџџџџџџџџЦ
Const_9Const*
_output_shapes	
:О*
dtype0	*
valueџBќ	О"№                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    	       	       	       	       	       	       	       	       	       	       
       
       
       
       
       
       
       
       
                                                                                                                                                                                                                                                                   Z
Shape_8ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
strided_slice_13StridedSliceShape_8:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 RZ
sub_8Substrided_slice_13:output:0sub_8/y:output:0*
T0	*
_output_shapes
: e
clip_by_value_5/MinimumMinimumConst_9:output:0	sub_8:z:0*
T0	*
_output_shapes	
:ОS
clip_by_value_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_5Maximumclip_by_value_5/Minimum:z:0clip_by_value_5/y:output:0*
T0	*
_output_shapes	
:ОQ
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B :Д

GatherV2_4GatherV2GatherV2_1:output:0clip_by_value_5:z:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџОg
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_14StridedSliceGatherV2_4:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџО*

begin_mask*
ellipsis_maskЧ
Const_10Const*
_output_shapes	
:О*
dtype0	*
valueџBќ	О"№                                                        	       
                                                                                                                       	       
                                                                                                                	       
                                                                                                         	       
                                                                                                  	       
                                                                                           	       
                                                                                    	       
                                                                             	       
                                                                      	       
                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                 Z
Shape_9ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
strided_slice_15StridedSliceShape_9:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
sub_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 RZ
sub_9Substrided_slice_15:output:0sub_9/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_6/MinimumMinimumConst_10:output:0	sub_9:z:0*
T0	*
_output_shapes	
:ОS
clip_by_value_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 R y
clip_by_value_6Maximumclip_by_value_6/Minimum:z:0clip_by_value_6/y:output:0*
T0	*
_output_shapes	
:ОQ
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B :Д

GatherV2_5GatherV2GatherV2_1:output:0clip_by_value_6:z:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџОg
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_16StridedSliceGatherV2_5:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџО*

begin_mask*
ellipsis_maskz
sub_10Substrided_slice_14:output:0strided_slice_16:output:0*
T0*,
_output_shapes
:џџџџџџџџџО`

norm_1/mulMul
sub_10:z:0
sub_10:z:0*
T0*,
_output_shapes
:џџџџџџџџџОo
norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_1/SumSumnorm_1/mul:z:0%norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџО*
	keep_dims(_
norm_1/SqrtSqrtnorm_1/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџО}
norm_1/SqueezeSqueezenorm_1/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџО*
squeeze_dims

џџџџџџџџџЬ
Const_11Const*
_output_shapes
:*
dtype0	*
valueB	"x                                                    	       
                                                   Y
Shape_10ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_17StridedSliceShape_10:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_11Substrided_slice_17:output:0sub_11/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_7/MinimumMinimumConst_11:output:0
sub_11:z:0*
T0	*
_output_shapes
:S
clip_by_value_7/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_7Maximumclip_by_value_7/Minimum:z:0clip_by_value_7/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_6GatherV2SelectV2:output:0clip_by_value_7:z:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџЬ
Const_12Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          	       
                                                        Y
Shape_11ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_18StridedSliceShape_11:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_12Substrided_slice_18:output:0sub_12/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_8/MinimumMinimumConst_12:output:0
sub_12:z:0*
T0	*
_output_shapes
:S
clip_by_value_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_8Maximumclip_by_value_8/Minimum:z:0clip_by_value_8/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_7GatherV2SelectV2:output:0clip_by_value_8:z:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџm
sub_13SubGatherV2_6:output:0GatherV2_7:output:0*
T0*+
_output_shapes
:џџџџџџџџџЬ
Const_13Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          
                                                               Y
Shape_12ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_19StridedSliceShape_12:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_14Substrided_slice_19:output:0sub_14/y:output:0*
T0	*
_output_shapes
: f
clip_by_value_9/MinimumMinimumConst_13:output:0
sub_14:z:0*
T0	*
_output_shapes
:S
clip_by_value_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 R x
clip_by_value_9Maximumclip_by_value_9/Minimum:z:0clip_by_value_9/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B :Б

GatherV2_8GatherV2SelectV2:output:0clip_by_value_9:z:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџЬ
Const_14Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          	       
                                                        Y
Shape_13ShapeSelectV2:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_20StridedSliceShape_13:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_15/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_15Substrided_slice_20:output:0sub_15/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_10/MinimumMinimumConst_14:output:0
sub_15:z:0*
T0	*
_output_shapes
:T
clip_by_value_10/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_10Maximumclip_by_value_10/Minimum:z:0clip_by_value_10/y:output:0*
T0	*
_output_shapes
:Q
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B :В

GatherV2_9GatherV2SelectV2:output:0clip_by_value_10:z:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџm
sub_16SubGatherV2_8:output:0GatherV2_9:output:0*
T0*+
_output_shapes
:џџџџџџџџџ_

norm_2/mulMul
sub_13:z:0
sub_13:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_2/SumSumnorm_2/mul:z:0%norm_2/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_2/SqrtSqrtnorm_2/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_2/SqueezeSqueezenorm_2/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџ_

norm_3/mulMul
sub_16:z:0
sub_16:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_3/SumSumnorm_3/mul:z:0%norm_3/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_3/SqrtSqrtnorm_3/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_3/SqueezeSqueezenorm_3/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџZ
mul_2Mul
sub_13:z:0
sub_16:z:0*
T0*+
_output_shapes
:џџџџџџџџџb
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџk
Sum_2Sum	mul_2:z:0 Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџo
	truediv_1RealDivSum_2:output:0norm_2/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
	truediv_2RealDivtruediv_1:z:0norm_3/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
Const_15Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@        	                            "              %       [
Shape_14ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_21StridedSliceShape_14:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_17/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_17Substrided_slice_21:output:0sub_17/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_11/MinimumMinimumConst_15:output:0
sub_17:z:0*
T0	*
_output_shapes
:T
clip_by_value_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_11Maximumclip_by_value_11/Minimum:z:0clip_by_value_11/y:output:0*
T0	*
_output_shapes
:R
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_10GatherV2GatherV2_1:output:0clip_by_value_11:z:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
Const_16Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@	                             "              %              [
Shape_15ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_22StridedSliceShape_15:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_18/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_18Substrided_slice_22:output:0sub_18/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_12/MinimumMinimumConst_16:output:0
sub_18:z:0*
T0	*
_output_shapes
:T
clip_by_value_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_12Maximumclip_by_value_12/Minimum:z:0clip_by_value_12/y:output:0*
T0	*
_output_shapes
:R
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_11GatherV2GatherV2_1:output:0clip_by_value_12:z:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџo
sub_19SubGatherV2_10:output:0GatherV2_11:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
Const_17Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@                      	              %              "       [
Shape_16ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_23StridedSliceShape_16:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_20/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_20Substrided_slice_23:output:0sub_20/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_13/MinimumMinimumConst_17:output:0
sub_20:z:0*
T0	*
_output_shapes
:T
clip_by_value_13/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_13Maximumclip_by_value_13/Minimum:z:0clip_by_value_13/y:output:0*
T0	*
_output_shapes
:R
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_12GatherV2GatherV2_1:output:0clip_by_value_13:z:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ
Const_18Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@	                             "              %              [
Shape_17ShapeGatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	`
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
strided_slice_24StridedSliceShape_17:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskJ
sub_21/yConst*
_output_shapes
: *
dtype0	*
value	B	 R\
sub_21Substrided_slice_24:output:0sub_21/y:output:0*
T0	*
_output_shapes
: g
clip_by_value_14/MinimumMinimumConst_18:output:0
sub_21:z:0*
T0	*
_output_shapes
:T
clip_by_value_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 R {
clip_by_value_14Maximumclip_by_value_14/Minimum:z:0clip_by_value_14/y:output:0*
T0	*
_output_shapes
:R
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B :Ж
GatherV2_13GatherV2GatherV2_1:output:0clip_by_value_14:z:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџo
sub_22SubGatherV2_12:output:0GatherV2_13:output:0*
T0*+
_output_shapes
:џџџџџџџџџ_

norm_4/mulMul
sub_19:z:0
sub_19:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_4/SumSumnorm_4/mul:z:0%norm_4/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_4/SqrtSqrtnorm_4/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_4/SqueezeSqueezenorm_4/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџ_

norm_5/mulMul
sub_22:z:0
sub_22:z:0*
T0*+
_output_shapes
:џџџџџџџџџo
norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ

norm_5/SumSumnorm_5/mul:z:0%norm_5/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(^
norm_5/SqrtSqrtnorm_5/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџ|
norm_5/SqueezeSqueezenorm_5/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџZ
mul_3Mul
sub_19:z:0
sub_22:z:0*
T0*+
_output_shapes
:џџџџџџџџџb
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџk
Sum_3Sum	mul_3:z:0 Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџo
	truediv_3RealDivSum_3:output:0norm_4/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџn
	truediv_4RealDivtruediv_3:z:0norm_5/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
concat_1ConcatV2GatherV2_1:output:0SelectV2:output:0concat_1/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџk
ReshapeReshapeconcat_1:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџP
IsNan_2IsNanReshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџJ

LogicalNot
LogicalNotIsNan_2:y:0*#
_output_shapes
:џџџџџџџџџR
boolean_mask/ShapeShapeReshape:output:0*
T0*
_output_shapes
:j
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ў
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:m
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: T
boolean_mask/Shape_1ShapeReshape:output:0*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskT
boolean_mask/Shape_2ShapeReshape:output:0*
T0*
_output_shapes
:l
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:n
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskn
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:Z
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : х
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:}
boolean_mask/ReshapeReshapeReshape:output:0boolean_mask/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџo
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
boolean_mask/Reshape_1ReshapeLogicalNot:y:0%boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџe
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
\
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : е
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџR
Const_19Const*
_output_shapes
:*
dtype0*
valueB: `
MeanMeanboolean_mask/GatherV2:output:0Const_19:output:0*
T0*
_output_shapes
: R
Const_20Const*
_output_shapes
:*
dtype0*
valueB: w
Mean_1Meanboolean_mask/GatherV2:output:0Const_20:output:0*
T0*
_output_shapes
:*
	keep_dims(l
sub_23Subboolean_mask/GatherV2:output:0Mean_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџJ
SquareSquare
sub_23:z:0*
T0*#
_output_shapes
:џџџџџџџџџR
Const_21Const*
_output_shapes
:*
dtype0*
valueB: L
Sum_4Sum
Square:y:0Const_21:output:0*
T0*
_output_shapes
: M
SizeSizeboolean_mask/GatherV2:output:0*
T0*
_output_shapes
: J
sub_24/yConst*
_output_shapes
: *
dtype0*
value	B :P
sub_24SubSize:output:0sub_24/y:output:0*
T0*
_output_shapes
: J
Cast_5Cast
sub_24:z:0*

DstT0*

SrcT0*
_output_shapes
: Q
	truediv_5RealDivSum_4:output:0
Cast_5:y:0*
T0*
_output_shapes
: <
SqrtSqrttruediv_5:z:0*
T0*
_output_shapes
: e
sub_25Subconcat_1:output:0Mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=`
	truediv_6RealDiv
sub_25:z:0Sqrt:y:0*
T0*+
_output_shapes
:џџџџџџџџџ=i
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_25StridedSlicetruediv_6:z:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*
end_maskh

zeros_like	ZerosLikestrided_slice_25:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=`
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
strided_slice_26StridedSlicetruediv_6:z:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*

begin_mask`
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_27StridedSlicetruediv_6:z:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*
end_masky
sub_26Substrided_slice_26:output:0strided_slice_27:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_2ConcatV2
sub_26:z:0zeros_like:y:0concat_2/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=L
NegNeg
sub_26:z:0*
T0*+
_output_shapes
:џџџџџџџџџ=O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2zeros_like:y:0Neg:y:0concat_3/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   x
flatten_1/ReshapeReshapetruediv_6:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗb
flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   
flatten_1/Reshape_1Reshapeconcat_2:output:0flatten_1/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗb
flatten_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   
flatten_1/Reshape_2Reshapeconcat_3:output:0flatten_1/Const_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗb
flatten_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"џџџџО   
flatten_1/Reshape_3Reshapenorm_1/Squeeze:output:0flatten_1/Const_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџОb
flatten_1/Const_4Const*
_output_shapes
:*
dtype0*
valueB"џџџџв   
flatten_1/Reshape_4Reshapenorm/Squeeze:output:0flatten_1/Const_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџвb
flatten_1/Const_5Const*
_output_shapes
:*
dtype0*
valueB"џџџџ   {
flatten_1/Reshape_5Reshapetruediv_4:z:0flatten_1/Const_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
flatten_1/Const_6Const*
_output_shapes
:*
dtype0*
valueB"џџџџ   {
flatten_1/Reshape_6Reshapetruediv_2:z:0flatten_1/Const_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
concat_4/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџИ
concat_4ConcatV2flatten_1/Reshape:output:0flatten_1/Reshape_1:output:0flatten_1/Reshape_2:output:0flatten_1/Reshape_3:output:0flatten_1/Reshape_4:output:0flatten_1/Reshape_5:output:0flatten_1/Reshape_6:output:0concat_4/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬV
IsNan_3IsNanconcat_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬQ
SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *    

SelectV2_1SelectV2IsNan_3:y:0SelectV2_1/t:output:0concat_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ`
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
strided_slice_28StridedSliceSelectV2_1:output:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџЬ*
new_axis_maskf
IdentityIdentitystrided_slice_28:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:Q M
,
_output_shapes
:џџџџџџџџџ

_user_specified_namepos: 

_output_shapes
:a
а
z
B__inference_model_1_layer_call_and_return_conditional_losses_97620

inputs
trans_preprocess_97616
identityх
 trans_preprocess/PartitionedCallPartitionedCallinputstrans_preprocess_97616*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_97567v
IdentityIdentity)trans_preprocess/PartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
с
m
__inference__traced_save_98076
file_prefix
savev2_const_1

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B к
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
Ц
L
#__inference_signature_wrapper_97636

inputs
unknown
identity
PartitionedCallPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_97151e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
а
z
B__inference_model_1_layer_call_and_return_conditional_losses_97601

inputs
trans_preprocess_97597
identityх
 trans_preprocess/PartitionedCallPartitionedCallinputstrans_preprocess_97597*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_97567v
IdentityIdentity)trans_preprocess/PartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
Ша
`
 __inference__wrapped_model_97151

inputs"
model_1_trans_preprocess_96766
identityT
model_1/trans_preprocess/ShapeShapeinputs*
T0*
_output_shapes
:v
,model_1/trans_preprocess/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model_1/trans_preprocess/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_1/trans_preprocess/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
&model_1/trans_preprocess/strided_sliceStridedSlice'model_1/trans_preprocess/Shape:output:05model_1/trans_preprocess/strided_slice/stack:output:07model_1/trans_preprocess/strided_slice/stack_1:output:07model_1/trans_preprocess/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model_1/trans_preprocess/add/yConst*
_output_shapes
: *
dtype0*
value	B : 
model_1/trans_preprocess/addAddV2/model_1/trans_preprocess/strided_slice:output:0'model_1/trans_preprocess/add/y:output:0*
T0*
_output_shapes
: `
model_1/trans_preprocess/ConstConst*
_output_shapes
: *
dtype0*
value	B :ak
 model_1/trans_preprocess/Const_1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџc
 model_1/trans_preprocess/Shape_1Const*
_output_shapes
: *
dtype0*
valueB c
 model_1/trans_preprocess/Shape_2Const*
_output_shapes
: *
dtype0*
valueB Ї
&model_1/trans_preprocess/BroadcastArgsBroadcastArgs)model_1/trans_preprocess/Shape_1:output:0)model_1/trans_preprocess/Shape_2:output:0*
_output_shapes
: c
 model_1/trans_preprocess/Shape_3Const*
_output_shapes
: *
dtype0*
valueB Ћ
(model_1/trans_preprocess/BroadcastArgs_1BroadcastArgs+model_1/trans_preprocess/BroadcastArgs:r0:0)model_1/trans_preprocess/Shape_3:output:0*
_output_shapes
: Ќ
$model_1/trans_preprocess/BroadcastToBroadcastTo'model_1/trans_preprocess/Const:output:0-model_1/trans_preprocess/BroadcastArgs_1:r0:0*
T0*
_output_shapes
: А
&model_1/trans_preprocess/BroadcastTo_1BroadcastTo)model_1/trans_preprocess/Const_1:output:0-model_1/trans_preprocess/BroadcastArgs_1:r0:0*
T0*
_output_shapes
: Ї
&model_1/trans_preprocess/BroadcastTo_2BroadcastTo model_1/trans_preprocess/add:z:0-model_1/trans_preprocess/BroadcastArgs_1:r0:0*
T0*
_output_shapes
: К
.model_1/trans_preprocess/clip_by_value/MinimumMinimum-model_1/trans_preprocess/BroadcastTo:output:0/model_1/trans_preprocess/BroadcastTo_2:output:0*
T0*
_output_shapes
: З
&model_1/trans_preprocess/clip_by_valueMaximum2model_1/trans_preprocess/clip_by_value/Minimum:z:0/model_1/trans_preprocess/BroadcastTo_1:output:0*
T0*
_output_shapes
: `
model_1/trans_preprocess/sub/yConst*
_output_shapes
: *
dtype0*
value	B :
model_1/trans_preprocess/subSub*model_1/trans_preprocess/clip_by_value:z:0'model_1/trans_preprocess/sub/y:output:0*
T0*
_output_shapes
: b
 model_1/trans_preprocess/Const_2Const*
_output_shapes
: *
dtype0*
value	B : b
 model_1/trans_preprocess/Const_3Const*
_output_shapes
: *
dtype0*
value	B :
.model_1/trans_preprocess/strided_slice_1/stackPack)model_1/trans_preprocess/Const_2:output:0*
N*
T0*
_output_shapes
:
0model_1/trans_preprocess/strided_slice_1/stack_1Pack model_1/trans_preprocess/sub:z:0*
N*
T0*
_output_shapes
:
0model_1/trans_preprocess/strided_slice_1/stack_2Pack)model_1/trans_preprocess/Const_3:output:0*
N*
T0*
_output_shapes
:д
(model_1/trans_preprocess/strided_slice_1StridedSlicemodel_1_trans_preprocess_967667model_1/trans_preprocess/strided_slice_1/stack:output:09model_1/trans_preprocess/strided_slice_1/stack_1:output:09model_1/trans_preprocess/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:џџџџџџџџџ*

begin_mask
model_1/trans_preprocess/CastCast/model_1/trans_preprocess/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ї
model_1/trans_preprocess/mulMul1model_1/trans_preprocess/strided_slice_1:output:0!model_1/trans_preprocess/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџb
 model_1/trans_preprocess/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :
model_1/trans_preprocess/sub_1Sub*model_1/trans_preprocess/clip_by_value:z:0)model_1/trans_preprocess/sub_1/y:output:0*
T0*
_output_shapes
: {
model_1/trans_preprocess/Cast_1Cast"model_1/trans_preprocess/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
:  
 model_1/trans_preprocess/truedivRealDiv model_1/trans_preprocess/mul:z:0#model_1/trans_preprocess/Cast_1:y:0*
T0*#
_output_shapes
:џџџџџџџџџ
model_1/trans_preprocess/Cast_2Cast$model_1/trans_preprocess/truediv:z:0*

DstT0*

SrcT0*#
_output_shapes
:џџџџџџџџџV
 model_1/trans_preprocess/Shape_4Shapeinputs*
T0*
_output_shapes
:x
.model_1/trans_preprocess/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/trans_preprocess/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/trans_preprocess/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(model_1/trans_preprocess/strided_slice_2StridedSlice)model_1/trans_preprocess/Shape_4:output:07model_1/trans_preprocess/strided_slice_2/stack:output:09model_1/trans_preprocess/strided_slice_2/stack_1:output:09model_1/trans_preprocess/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model_1/trans_preprocess/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :Є
model_1/trans_preprocess/sub_2Sub1model_1/trans_preprocess/strided_slice_2:output:0)model_1/trans_preprocess/sub_2/y:output:0*
T0*
_output_shapes
: В
0model_1/trans_preprocess/clip_by_value_1/MinimumMinimum#model_1/trans_preprocess/Cast_2:y:0"model_1/trans_preprocess/sub_2:z:0*
T0*#
_output_shapes
:џџџџџџџџџl
*model_1/trans_preprocess/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ь
(model_1/trans_preprocess/clip_by_value_1Maximum4model_1/trans_preprocess/clip_by_value_1/Minimum:z:03model_1/trans_preprocess/clip_by_value_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџh
&model_1/trans_preprocess/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ю
!model_1/trans_preprocess/GatherV2GatherV2inputs,model_1/trans_preprocess/clip_by_value_1:z:0/model_1/trans_preprocess/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*,
_output_shapes
:џџџџџџџџџ­
 model_1/trans_preprocess/Const_4Const*
_output_shapes
:(*
dtype0	*и
valueЮBЫ	("Р                                                     #      8      7      6            4      =            >      D      :            A      w      %       '       (       Й       =       R       Q       P       П       N       W       В       X       _       T       Е       [              
 model_1/trans_preprocess/Shape_5Shape*model_1/trans_preprocess/GatherV2:output:0*
T0*
_output_shapes
:*
out_type0	x
.model_1/trans_preprocess/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0model_1/trans_preprocess/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/trans_preprocess/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(model_1/trans_preprocess/strided_slice_3StridedSlice)model_1/trans_preprocess/Shape_5:output:07model_1/trans_preprocess/strided_slice_3/stack:output:09model_1/trans_preprocess/strided_slice_3/stack_1:output:09model_1/trans_preprocess/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 model_1/trans_preprocess/sub_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЄ
model_1/trans_preprocess/sub_3Sub1model_1/trans_preprocess/strided_slice_3:output:0)model_1/trans_preprocess/sub_3/y:output:0*
T0	*
_output_shapes
: Џ
0model_1/trans_preprocess/clip_by_value_2/MinimumMinimum)model_1/trans_preprocess/Const_4:output:0"model_1/trans_preprocess/sub_3:z:0*
T0	*
_output_shapes
:(l
*model_1/trans_preprocess/clip_by_value_2/yConst*
_output_shapes
: *
dtype0	*
value	B	 R У
(model_1/trans_preprocess/clip_by_value_2Maximum4model_1/trans_preprocess/clip_by_value_2/Minimum:z:03model_1/trans_preprocess/clip_by_value_2/y:output:0*
T0	*
_output_shapes
:(j
(model_1/trans_preprocess/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_1GatherV2*model_1/trans_preprocess/GatherV2:output:0,model_1/trans_preprocess/clip_by_value_2:z:01model_1/trans_preprocess/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџ(
.model_1/trans_preprocess/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    д  
0model_1/trans_preprocess/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    щ  
0model_1/trans_preprocess/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
(model_1/trans_preprocess/strided_slice_4StridedSlice*model_1/trans_preprocess/GatherV2:output:07model_1/trans_preprocess/strided_slice_4/stack:output:09model_1/trans_preprocess/strided_slice_4/stack_1:output:09model_1/trans_preprocess/strided_slice_4/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask
.model_1/trans_preprocess/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    
  
0model_1/trans_preprocess/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      
0model_1/trans_preprocess/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ј
(model_1/trans_preprocess/strided_slice_5StridedSlice*model_1/trans_preprocess/GatherV2:output:07model_1/trans_preprocess/strided_slice_5/stack:output:09model_1/trans_preprocess/strided_slice_5/stack_1:output:09model_1/trans_preprocess/strided_slice_5/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask
.model_1/trans_preprocess/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
0model_1/trans_preprocess/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"          
0model_1/trans_preprocess/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         њ
(model_1/trans_preprocess/strided_slice_6StridedSlice,model_1/trans_preprocess/GatherV2_1:output:07model_1/trans_preprocess/strided_slice_6/stack:output:09model_1/trans_preprocess/strided_slice_6/stack_1:output:09model_1/trans_preprocess/strided_slice_6/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maske
 model_1/trans_preprocess/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Й
model_1/trans_preprocess/mul_1Mul)model_1/trans_preprocess/mul_1/x:output:01model_1/trans_preprocess/strided_slice_6:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
.model_1/trans_preprocess/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0model_1/trans_preprocess/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0model_1/trans_preprocess/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ђ
(model_1/trans_preprocess/strided_slice_7StridedSlice1model_1/trans_preprocess/strided_slice_5:output:07model_1/trans_preprocess/strided_slice_7/stack:output:09model_1/trans_preprocess/strided_slice_7/stack_1:output:09model_1/trans_preprocess/strided_slice_7/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
ellipsis_maskВ
model_1/trans_preprocess/sub_4Sub"model_1/trans_preprocess/mul_1:z:01model_1/trans_preprocess/strided_slice_7:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
.model_1/trans_preprocess/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0model_1/trans_preprocess/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0model_1/trans_preprocess/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
(model_1/trans_preprocess/strided_slice_8StridedSlice1model_1/trans_preprocess/strided_slice_5:output:07model_1/trans_preprocess/strided_slice_8/stack:output:09model_1/trans_preprocess/strided_slice_8/stack_1:output:09model_1/trans_preprocess/strided_slice_8/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ*
ellipsis_mask*
end_masko
$model_1/trans_preprocess/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ№
model_1/trans_preprocess/concatConcatV2"model_1/trans_preprocess/sub_4:z:01model_1/trans_preprocess/strided_slice_8:output:0-model_1/trans_preprocess/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
model_1/trans_preprocess/IsNanIsNan1model_1/trans_preprocess/strided_slice_4:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
model_1/trans_preprocess/Cast_3Cast"model_1/trans_preprocess/IsNan:y:0*

DstT0	*

SrcT0
*+
_output_shapes
:џџџџџџџџџu
 model_1/trans_preprocess/Const_5Const*
_output_shapes
:*
dtype0*!
valueB"          
model_1/trans_preprocess/SumSum#model_1/trans_preprocess/Cast_3:y:0)model_1/trans_preprocess/Const_5:output:0*
T0	*
_output_shapes
: 
 model_1/trans_preprocess/IsNan_1IsNan(model_1/trans_preprocess/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
model_1/trans_preprocess/Cast_4Cast$model_1/trans_preprocess/IsNan_1:y:0*

DstT0	*

SrcT0
*+
_output_shapes
:џџџџџџџџџu
 model_1/trans_preprocess/Const_6Const*
_output_shapes
:*
dtype0*!
valueB"          
model_1/trans_preprocess/Sum_1Sum#model_1/trans_preprocess/Cast_4:y:0)model_1/trans_preprocess/Const_6:output:0*
T0	*
_output_shapes
: 
model_1/trans_preprocess/LessLess%model_1/trans_preprocess/Sum:output:0'model_1/trans_preprocess/Sum_1:output:0*
T0	*
_output_shapes
: у
!model_1/trans_preprocess/SelectV2SelectV2!model_1/trans_preprocess/Less:z:01model_1/trans_preprocess/strided_slice_4:output:0(model_1/trans_preprocess/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџџ
 model_1/trans_preprocess/Const_7Const*
_output_shapes	
:в*
dtype0	*Љ
valueB	в"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    	       	       	       	       	       	       	       	       	       	       	       
       
       
       
       
       
       
       
       
       
                                                                                                                                                                                                                                                                                                                                  
 model_1/trans_preprocess/Shape_6Shape*model_1/trans_preprocess/SelectV2:output:0*
T0*
_output_shapes
:*
out_type0	x
.model_1/trans_preprocess/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0model_1/trans_preprocess/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/trans_preprocess/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(model_1/trans_preprocess/strided_slice_9StridedSlice)model_1/trans_preprocess/Shape_6:output:07model_1/trans_preprocess/strided_slice_9/stack:output:09model_1/trans_preprocess/strided_slice_9/stack_1:output:09model_1/trans_preprocess/strided_slice_9/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 model_1/trans_preprocess/sub_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЄ
model_1/trans_preprocess/sub_5Sub1model_1/trans_preprocess/strided_slice_9:output:0)model_1/trans_preprocess/sub_5/y:output:0*
T0	*
_output_shapes
: А
0model_1/trans_preprocess/clip_by_value_3/MinimumMinimum)model_1/trans_preprocess/Const_7:output:0"model_1/trans_preprocess/sub_5:z:0*
T0	*
_output_shapes	
:вl
*model_1/trans_preprocess/clip_by_value_3/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ф
(model_1/trans_preprocess/clip_by_value_3Maximum4model_1/trans_preprocess/clip_by_value_3/Minimum:z:03model_1/trans_preprocess/clip_by_value_3/y:output:0*
T0	*
_output_shapes	
:вj
(model_1/trans_preprocess/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_2GatherV2*model_1/trans_preprocess/SelectV2:output:0,model_1/trans_preprocess/clip_by_value_3:z:01model_1/trans_preprocess/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџв
/model_1/trans_preprocess/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1model_1/trans_preprocess/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1model_1/trans_preprocess/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)model_1/trans_preprocess/strided_slice_10StridedSlice,model_1/trans_preprocess/GatherV2_2:output:08model_1/trans_preprocess/strided_slice_10/stack:output:0:model_1/trans_preprocess/strided_slice_10/stack_1:output:0:model_1/trans_preprocess/strided_slice_10/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџв*

begin_mask*
ellipsis_maskџ
 model_1/trans_preprocess/Const_8Const*
_output_shapes	
:в*
dtype0	*Љ
valueB	в"                                                        	       
                                                                                                                              	       
                                                                                                                       	       
                                                                                                                	       
                                                                                                         	       
                                                                                                  	       
                                                                                           	       
                                                                                    	       
                                                                             	       
                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
 model_1/trans_preprocess/Shape_7Shape*model_1/trans_preprocess/SelectV2:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
)model_1/trans_preprocess/strided_slice_11StridedSlice)model_1/trans_preprocess/Shape_7:output:08model_1/trans_preprocess/strided_slice_11/stack:output:0:model_1/trans_preprocess/strided_slice_11/stack_1:output:0:model_1/trans_preprocess/strided_slice_11/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 model_1/trans_preprocess/sub_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЅ
model_1/trans_preprocess/sub_6Sub2model_1/trans_preprocess/strided_slice_11:output:0)model_1/trans_preprocess/sub_6/y:output:0*
T0	*
_output_shapes
: А
0model_1/trans_preprocess/clip_by_value_4/MinimumMinimum)model_1/trans_preprocess/Const_8:output:0"model_1/trans_preprocess/sub_6:z:0*
T0	*
_output_shapes	
:вl
*model_1/trans_preprocess/clip_by_value_4/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ф
(model_1/trans_preprocess/clip_by_value_4Maximum4model_1/trans_preprocess/clip_by_value_4/Minimum:z:03model_1/trans_preprocess/clip_by_value_4/y:output:0*
T0	*
_output_shapes	
:вj
(model_1/trans_preprocess/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_3GatherV2*model_1/trans_preprocess/SelectV2:output:0,model_1/trans_preprocess/clip_by_value_4:z:01model_1/trans_preprocess/GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџв
/model_1/trans_preprocess/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1model_1/trans_preprocess/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1model_1/trans_preprocess/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)model_1/trans_preprocess/strided_slice_12StridedSlice,model_1/trans_preprocess/GatherV2_3:output:08model_1/trans_preprocess/strided_slice_12/stack:output:0:model_1/trans_preprocess/strided_slice_12/stack_1:output:0:model_1/trans_preprocess/strided_slice_12/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџв*

begin_mask*
ellipsis_maskФ
model_1/trans_preprocess/sub_7Sub2model_1/trans_preprocess/strided_slice_10:output:02model_1/trans_preprocess/strided_slice_12:output:0*
T0*,
_output_shapes
:џџџџџџџџџвЇ
!model_1/trans_preprocess/norm/mulMul"model_1/trans_preprocess/sub_7:z:0"model_1/trans_preprocess/sub_7:z:0*
T0*,
_output_shapes
:џџџџџџџџџв
3model_1/trans_preprocess/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџе
!model_1/trans_preprocess/norm/SumSum%model_1/trans_preprocess/norm/mul:z:0<model_1/trans_preprocess/norm/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџв*
	keep_dims(
"model_1/trans_preprocess/norm/SqrtSqrt*model_1/trans_preprocess/norm/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџвЋ
%model_1/trans_preprocess/norm/SqueezeSqueeze&model_1/trans_preprocess/norm/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџв*
squeeze_dims

џџџџџџџџџп
 model_1/trans_preprocess/Const_9Const*
_output_shapes	
:О*
dtype0	*
valueџBќ	О"№                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    	       	       	       	       	       	       	       	       	       	       
       
       
       
       
       
       
       
       
                                                                                                                                                                                                                                                                   
 model_1/trans_preprocess/Shape_8Shape,model_1/trans_preprocess/GatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
)model_1/trans_preprocess/strided_slice_13StridedSlice)model_1/trans_preprocess/Shape_8:output:08model_1/trans_preprocess/strided_slice_13/stack:output:0:model_1/trans_preprocess/strided_slice_13/stack_1:output:0:model_1/trans_preprocess/strided_slice_13/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 model_1/trans_preprocess/sub_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЅ
model_1/trans_preprocess/sub_8Sub2model_1/trans_preprocess/strided_slice_13:output:0)model_1/trans_preprocess/sub_8/y:output:0*
T0	*
_output_shapes
: А
0model_1/trans_preprocess/clip_by_value_5/MinimumMinimum)model_1/trans_preprocess/Const_9:output:0"model_1/trans_preprocess/sub_8:z:0*
T0	*
_output_shapes	
:Оl
*model_1/trans_preprocess/clip_by_value_5/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ф
(model_1/trans_preprocess/clip_by_value_5Maximum4model_1/trans_preprocess/clip_by_value_5/Minimum:z:03model_1/trans_preprocess/clip_by_value_5/y:output:0*
T0	*
_output_shapes	
:Оj
(model_1/trans_preprocess/GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_4GatherV2,model_1/trans_preprocess/GatherV2_1:output:0,model_1/trans_preprocess/clip_by_value_5:z:01model_1/trans_preprocess/GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџО
/model_1/trans_preprocess/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1model_1/trans_preprocess/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1model_1/trans_preprocess/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)model_1/trans_preprocess/strided_slice_14StridedSlice,model_1/trans_preprocess/GatherV2_4:output:08model_1/trans_preprocess/strided_slice_14/stack:output:0:model_1/trans_preprocess/strided_slice_14/stack_1:output:0:model_1/trans_preprocess/strided_slice_14/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџО*

begin_mask*
ellipsis_maskр
!model_1/trans_preprocess/Const_10Const*
_output_shapes	
:О*
dtype0	*
valueџBќ	О"№                                                        	       
                                                                                                                       	       
                                                                                                                	       
                                                                                                         	       
                                                                                                  	       
                                                                                           	       
                                                                                    	       
                                                                             	       
                                                                      	       
                                                                      
                                                                                                                                                                                                                                                                                                                                                                                                 
 model_1/trans_preprocess/Shape_9Shape,model_1/trans_preprocess/GatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
)model_1/trans_preprocess/strided_slice_15StridedSlice)model_1/trans_preprocess/Shape_9:output:08model_1/trans_preprocess/strided_slice_15/stack:output:0:model_1/trans_preprocess/strided_slice_15/stack_1:output:0:model_1/trans_preprocess/strided_slice_15/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 model_1/trans_preprocess/sub_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЅ
model_1/trans_preprocess/sub_9Sub2model_1/trans_preprocess/strided_slice_15:output:0)model_1/trans_preprocess/sub_9/y:output:0*
T0	*
_output_shapes
: Б
0model_1/trans_preprocess/clip_by_value_6/MinimumMinimum*model_1/trans_preprocess/Const_10:output:0"model_1/trans_preprocess/sub_9:z:0*
T0	*
_output_shapes	
:Оl
*model_1/trans_preprocess/clip_by_value_6/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ф
(model_1/trans_preprocess/clip_by_value_6Maximum4model_1/trans_preprocess/clip_by_value_6/Minimum:z:03model_1/trans_preprocess/clip_by_value_6/y:output:0*
T0	*
_output_shapes	
:Оj
(model_1/trans_preprocess/GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_5GatherV2,model_1/trans_preprocess/GatherV2_1:output:0,model_1/trans_preprocess/clip_by_value_6:z:01model_1/trans_preprocess/GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*,
_output_shapes
:џџџџџџџџџО
/model_1/trans_preprocess/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1model_1/trans_preprocess/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1model_1/trans_preprocess/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)model_1/trans_preprocess/strided_slice_16StridedSlice,model_1/trans_preprocess/GatherV2_5:output:08model_1/trans_preprocess/strided_slice_16/stack:output:0:model_1/trans_preprocess/strided_slice_16/stack_1:output:0:model_1/trans_preprocess/strided_slice_16/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџО*

begin_mask*
ellipsis_maskХ
model_1/trans_preprocess/sub_10Sub2model_1/trans_preprocess/strided_slice_14:output:02model_1/trans_preprocess/strided_slice_16:output:0*
T0*,
_output_shapes
:џџџџџџџџџОЋ
#model_1/trans_preprocess/norm_1/mulMul#model_1/trans_preprocess/sub_10:z:0#model_1/trans_preprocess/sub_10:z:0*
T0*,
_output_shapes
:џџџџџџџџџО
5model_1/trans_preprocess/norm_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџл
#model_1/trans_preprocess/norm_1/SumSum'model_1/trans_preprocess/norm_1/mul:z:0>model_1/trans_preprocess/norm_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:џџџџџџџџџО*
	keep_dims(
$model_1/trans_preprocess/norm_1/SqrtSqrt,model_1/trans_preprocess/norm_1/Sum:output:0*
T0*,
_output_shapes
:џџџџџџџџџОЏ
'model_1/trans_preprocess/norm_1/SqueezeSqueeze(model_1/trans_preprocess/norm_1/Sqrt:y:0*
T0*(
_output_shapes
:џџџџџџџџџО*
squeeze_dims

џџџџџџџџџх
!model_1/trans_preprocess/Const_11Const*
_output_shapes
:*
dtype0	*
valueB	"x                                                    	       
                                                   
!model_1/trans_preprocess/Shape_10Shape*model_1/trans_preprocess/SelectV2:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_17StridedSlice*model_1/trans_preprocess/Shape_10:output:08model_1/trans_preprocess/strided_slice_17/stack:output:0:model_1/trans_preprocess/strided_slice_17/stack_1:output:0:model_1/trans_preprocess/strided_slice_17/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_11Sub2model_1/trans_preprocess/strided_slice_17:output:0*model_1/trans_preprocess/sub_11/y:output:0*
T0	*
_output_shapes
: Б
0model_1/trans_preprocess/clip_by_value_7/MinimumMinimum*model_1/trans_preprocess/Const_11:output:0#model_1/trans_preprocess/sub_11:z:0*
T0	*
_output_shapes
:l
*model_1/trans_preprocess/clip_by_value_7/yConst*
_output_shapes
: *
dtype0	*
value	B	 R У
(model_1/trans_preprocess/clip_by_value_7Maximum4model_1/trans_preprocess/clip_by_value_7/Minimum:z:03model_1/trans_preprocess/clip_by_value_7/y:output:0*
T0	*
_output_shapes
:j
(model_1/trans_preprocess/GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_6GatherV2*model_1/trans_preprocess/SelectV2:output:0,model_1/trans_preprocess/clip_by_value_7:z:01model_1/trans_preprocess/GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџх
!model_1/trans_preprocess/Const_12Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          	       
                                                        
!model_1/trans_preprocess/Shape_11Shape*model_1/trans_preprocess/SelectV2:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_18StridedSlice*model_1/trans_preprocess/Shape_11:output:08model_1/trans_preprocess/strided_slice_18/stack:output:0:model_1/trans_preprocess/strided_slice_18/stack_1:output:0:model_1/trans_preprocess/strided_slice_18/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_12Sub2model_1/trans_preprocess/strided_slice_18:output:0*model_1/trans_preprocess/sub_12/y:output:0*
T0	*
_output_shapes
: Б
0model_1/trans_preprocess/clip_by_value_8/MinimumMinimum*model_1/trans_preprocess/Const_12:output:0#model_1/trans_preprocess/sub_12:z:0*
T0	*
_output_shapes
:l
*model_1/trans_preprocess/clip_by_value_8/yConst*
_output_shapes
: *
dtype0	*
value	B	 R У
(model_1/trans_preprocess/clip_by_value_8Maximum4model_1/trans_preprocess/clip_by_value_8/Minimum:z:03model_1/trans_preprocess/clip_by_value_8/y:output:0*
T0	*
_output_shapes
:j
(model_1/trans_preprocess/GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_7GatherV2*model_1/trans_preprocess/SelectV2:output:0,model_1/trans_preprocess/clip_by_value_8:z:01model_1/trans_preprocess/GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџИ
model_1/trans_preprocess/sub_13Sub,model_1/trans_preprocess/GatherV2_6:output:0,model_1/trans_preprocess/GatherV2_7:output:0*
T0*+
_output_shapes
:џџџџџџџџџх
!model_1/trans_preprocess/Const_13Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          
                                                               
!model_1/trans_preprocess/Shape_12Shape*model_1/trans_preprocess/SelectV2:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_19StridedSlice*model_1/trans_preprocess/Shape_12:output:08model_1/trans_preprocess/strided_slice_19/stack:output:0:model_1/trans_preprocess/strided_slice_19/stack_1:output:0:model_1/trans_preprocess/strided_slice_19/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_14Sub2model_1/trans_preprocess/strided_slice_19:output:0*model_1/trans_preprocess/sub_14/y:output:0*
T0	*
_output_shapes
: Б
0model_1/trans_preprocess/clip_by_value_9/MinimumMinimum*model_1/trans_preprocess/Const_13:output:0#model_1/trans_preprocess/sub_14:z:0*
T0	*
_output_shapes
:l
*model_1/trans_preprocess/clip_by_value_9/yConst*
_output_shapes
: *
dtype0	*
value	B	 R У
(model_1/trans_preprocess/clip_by_value_9Maximum4model_1/trans_preprocess/clip_by_value_9/Minimum:z:03model_1/trans_preprocess/clip_by_value_9/y:output:0*
T0	*
_output_shapes
:j
(model_1/trans_preprocess/GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_8GatherV2*model_1/trans_preprocess/SelectV2:output:0,model_1/trans_preprocess/clip_by_value_9:z:01model_1/trans_preprocess/GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџх
!model_1/trans_preprocess/Const_14Const*
_output_shapes
:*
dtype0	*
valueB	"x                                          	       
                                                        
!model_1/trans_preprocess/Shape_13Shape*model_1/trans_preprocess/SelectV2:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_20StridedSlice*model_1/trans_preprocess/Shape_13:output:08model_1/trans_preprocess/strided_slice_20/stack:output:0:model_1/trans_preprocess/strided_slice_20/stack_1:output:0:model_1/trans_preprocess/strided_slice_20/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_15/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_15Sub2model_1/trans_preprocess/strided_slice_20:output:0*model_1/trans_preprocess/sub_15/y:output:0*
T0	*
_output_shapes
: В
1model_1/trans_preprocess/clip_by_value_10/MinimumMinimum*model_1/trans_preprocess/Const_14:output:0#model_1/trans_preprocess/sub_15:z:0*
T0	*
_output_shapes
:m
+model_1/trans_preprocess/clip_by_value_10/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ц
)model_1/trans_preprocess/clip_by_value_10Maximum5model_1/trans_preprocess/clip_by_value_10/Minimum:z:04model_1/trans_preprocess/clip_by_value_10/y:output:0*
T0	*
_output_shapes
:j
(model_1/trans_preprocess/GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/trans_preprocess/GatherV2_9GatherV2*model_1/trans_preprocess/SelectV2:output:0-model_1/trans_preprocess/clip_by_value_10:z:01model_1/trans_preprocess/GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџИ
model_1/trans_preprocess/sub_16Sub,model_1/trans_preprocess/GatherV2_8:output:0,model_1/trans_preprocess/GatherV2_9:output:0*
T0*+
_output_shapes
:џџџџџџџџџЊ
#model_1/trans_preprocess/norm_2/mulMul#model_1/trans_preprocess/sub_13:z:0#model_1/trans_preprocess/sub_13:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
5model_1/trans_preprocess/norm_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџк
#model_1/trans_preprocess/norm_2/SumSum'model_1/trans_preprocess/norm_2/mul:z:0>model_1/trans_preprocess/norm_2/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(
$model_1/trans_preprocess/norm_2/SqrtSqrt,model_1/trans_preprocess/norm_2/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџЎ
'model_1/trans_preprocess/norm_2/SqueezeSqueeze(model_1/trans_preprocess/norm_2/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџЊ
#model_1/trans_preprocess/norm_3/mulMul#model_1/trans_preprocess/sub_16:z:0#model_1/trans_preprocess/sub_16:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
5model_1/trans_preprocess/norm_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџк
#model_1/trans_preprocess/norm_3/SumSum'model_1/trans_preprocess/norm_3/mul:z:0>model_1/trans_preprocess/norm_3/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(
$model_1/trans_preprocess/norm_3/SqrtSqrt,model_1/trans_preprocess/norm_3/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџЎ
'model_1/trans_preprocess/norm_3/SqueezeSqueeze(model_1/trans_preprocess/norm_3/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџЅ
model_1/trans_preprocess/mul_2Mul#model_1/trans_preprocess/sub_13:z:0#model_1/trans_preprocess/sub_16:z:0*
T0*+
_output_shapes
:џџџџџџџџџ{
0model_1/trans_preprocess/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЖ
model_1/trans_preprocess/Sum_2Sum"model_1/trans_preprocess/mul_2:z:09model_1/trans_preprocess/Sum_2/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџК
"model_1/trans_preprocess/truediv_1RealDiv'model_1/trans_preprocess/Sum_2:output:00model_1/trans_preprocess/norm_2/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
"model_1/trans_preprocess/truediv_2RealDiv&model_1/trans_preprocess/truediv_1:z:00model_1/trans_preprocess/norm_3/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
!model_1/trans_preprocess/Const_15Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@        	                            "              %       
!model_1/trans_preprocess/Shape_14Shape,model_1/trans_preprocess/GatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_21StridedSlice*model_1/trans_preprocess/Shape_14:output:08model_1/trans_preprocess/strided_slice_21/stack:output:0:model_1/trans_preprocess/strided_slice_21/stack_1:output:0:model_1/trans_preprocess/strided_slice_21/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_17/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_17Sub2model_1/trans_preprocess/strided_slice_21:output:0*model_1/trans_preprocess/sub_17/y:output:0*
T0	*
_output_shapes
: В
1model_1/trans_preprocess/clip_by_value_11/MinimumMinimum*model_1/trans_preprocess/Const_15:output:0#model_1/trans_preprocess/sub_17:z:0*
T0	*
_output_shapes
:m
+model_1/trans_preprocess/clip_by_value_11/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ц
)model_1/trans_preprocess/clip_by_value_11Maximum5model_1/trans_preprocess/clip_by_value_11/Minimum:z:04model_1/trans_preprocess/clip_by_value_11/y:output:0*
T0	*
_output_shapes
:k
)model_1/trans_preprocess/GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B :
$model_1/trans_preprocess/GatherV2_10GatherV2,model_1/trans_preprocess/GatherV2_1:output:0-model_1/trans_preprocess/clip_by_value_11:z:02model_1/trans_preprocess/GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџЊ
!model_1/trans_preprocess/Const_16Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@	                             "              %              
!model_1/trans_preprocess/Shape_15Shape,model_1/trans_preprocess/GatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_22StridedSlice*model_1/trans_preprocess/Shape_15:output:08model_1/trans_preprocess/strided_slice_22/stack:output:0:model_1/trans_preprocess/strided_slice_22/stack_1:output:0:model_1/trans_preprocess/strided_slice_22/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_18/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_18Sub2model_1/trans_preprocess/strided_slice_22:output:0*model_1/trans_preprocess/sub_18/y:output:0*
T0	*
_output_shapes
: В
1model_1/trans_preprocess/clip_by_value_12/MinimumMinimum*model_1/trans_preprocess/Const_16:output:0#model_1/trans_preprocess/sub_18:z:0*
T0	*
_output_shapes
:m
+model_1/trans_preprocess/clip_by_value_12/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ц
)model_1/trans_preprocess/clip_by_value_12Maximum5model_1/trans_preprocess/clip_by_value_12/Minimum:z:04model_1/trans_preprocess/clip_by_value_12/y:output:0*
T0	*
_output_shapes
:k
)model_1/trans_preprocess/GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B :
$model_1/trans_preprocess/GatherV2_11GatherV2,model_1/trans_preprocess/GatherV2_1:output:0-model_1/trans_preprocess/clip_by_value_12:z:02model_1/trans_preprocess/GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџК
model_1/trans_preprocess/sub_19Sub-model_1/trans_preprocess/GatherV2_10:output:0-model_1/trans_preprocess/GatherV2_11:output:0*
T0*+
_output_shapes
:џџџџџџџџџЊ
!model_1/trans_preprocess/Const_17Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@                      	              %              "       
!model_1/trans_preprocess/Shape_16Shape,model_1/trans_preprocess/GatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_23StridedSlice*model_1/trans_preprocess/Shape_16:output:08model_1/trans_preprocess/strided_slice_23/stack:output:0:model_1/trans_preprocess/strided_slice_23/stack_1:output:0:model_1/trans_preprocess/strided_slice_23/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_20/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_20Sub2model_1/trans_preprocess/strided_slice_23:output:0*model_1/trans_preprocess/sub_20/y:output:0*
T0	*
_output_shapes
: В
1model_1/trans_preprocess/clip_by_value_13/MinimumMinimum*model_1/trans_preprocess/Const_17:output:0#model_1/trans_preprocess/sub_20:z:0*
T0	*
_output_shapes
:m
+model_1/trans_preprocess/clip_by_value_13/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ц
)model_1/trans_preprocess/clip_by_value_13Maximum5model_1/trans_preprocess/clip_by_value_13/Minimum:z:04model_1/trans_preprocess/clip_by_value_13/y:output:0*
T0	*
_output_shapes
:k
)model_1/trans_preprocess/GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B :
$model_1/trans_preprocess/GatherV2_12GatherV2,model_1/trans_preprocess/GatherV2_1:output:0-model_1/trans_preprocess/clip_by_value_13:z:02model_1/trans_preprocess/GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџЊ
!model_1/trans_preprocess/Const_18Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@	                             "              %              
!model_1/trans_preprocess/Shape_17Shape,model_1/trans_preprocess/GatherV2_1:output:0*
T0*
_output_shapes
:*
out_type0	y
/model_1/trans_preprocess/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)model_1/trans_preprocess/strided_slice_24StridedSlice*model_1/trans_preprocess/Shape_17:output:08model_1/trans_preprocess/strided_slice_24/stack:output:0:model_1/trans_preprocess/strided_slice_24/stack_1:output:0:model_1/trans_preprocess/strided_slice_24/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskc
!model_1/trans_preprocess/sub_21/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЇ
model_1/trans_preprocess/sub_21Sub2model_1/trans_preprocess/strided_slice_24:output:0*model_1/trans_preprocess/sub_21/y:output:0*
T0	*
_output_shapes
: В
1model_1/trans_preprocess/clip_by_value_14/MinimumMinimum*model_1/trans_preprocess/Const_18:output:0#model_1/trans_preprocess/sub_21:z:0*
T0	*
_output_shapes
:m
+model_1/trans_preprocess/clip_by_value_14/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Ц
)model_1/trans_preprocess/clip_by_value_14Maximum5model_1/trans_preprocess/clip_by_value_14/Minimum:z:04model_1/trans_preprocess/clip_by_value_14/y:output:0*
T0	*
_output_shapes
:k
)model_1/trans_preprocess/GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B :
$model_1/trans_preprocess/GatherV2_13GatherV2,model_1/trans_preprocess/GatherV2_1:output:0-model_1/trans_preprocess/clip_by_value_14:z:02model_1/trans_preprocess/GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*+
_output_shapes
:џџџџџџџџџК
model_1/trans_preprocess/sub_22Sub-model_1/trans_preprocess/GatherV2_12:output:0-model_1/trans_preprocess/GatherV2_13:output:0*
T0*+
_output_shapes
:џџџџџџџџџЊ
#model_1/trans_preprocess/norm_4/mulMul#model_1/trans_preprocess/sub_19:z:0#model_1/trans_preprocess/sub_19:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
5model_1/trans_preprocess/norm_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџк
#model_1/trans_preprocess/norm_4/SumSum'model_1/trans_preprocess/norm_4/mul:z:0>model_1/trans_preprocess/norm_4/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(
$model_1/trans_preprocess/norm_4/SqrtSqrt,model_1/trans_preprocess/norm_4/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџЎ
'model_1/trans_preprocess/norm_4/SqueezeSqueeze(model_1/trans_preprocess/norm_4/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџЊ
#model_1/trans_preprocess/norm_5/mulMul#model_1/trans_preprocess/sub_22:z:0#model_1/trans_preprocess/sub_22:z:0*
T0*+
_output_shapes
:џџџџџџџџџ
5model_1/trans_preprocess/norm_5/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџк
#model_1/trans_preprocess/norm_5/SumSum'model_1/trans_preprocess/norm_5/mul:z:0>model_1/trans_preprocess/norm_5/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:џџџџџџџџџ*
	keep_dims(
$model_1/trans_preprocess/norm_5/SqrtSqrt,model_1/trans_preprocess/norm_5/Sum:output:0*
T0*+
_output_shapes
:џџџџџџџџџЎ
'model_1/trans_preprocess/norm_5/SqueezeSqueeze(model_1/trans_preprocess/norm_5/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims

џџџџџџџџџЅ
model_1/trans_preprocess/mul_3Mul#model_1/trans_preprocess/sub_19:z:0#model_1/trans_preprocess/sub_22:z:0*
T0*+
_output_shapes
:џџџџџџџџџ{
0model_1/trans_preprocess/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЖ
model_1/trans_preprocess/Sum_3Sum"model_1/trans_preprocess/mul_3:z:09model_1/trans_preprocess/Sum_3/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџК
"model_1/trans_preprocess/truediv_3RealDiv'model_1/trans_preprocess/Sum_3:output:00model_1/trans_preprocess/norm_4/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
"model_1/trans_preprocess/truediv_4RealDiv&model_1/trans_preprocess/truediv_3:z:00model_1/trans_preprocess/norm_5/Squeeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
&model_1/trans_preprocess/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :ї
!model_1/trans_preprocess/concat_1ConcatV2,model_1/trans_preprocess/GatherV2_1:output:0*model_1/trans_preprocess/SelectV2:output:0/model_1/trans_preprocess/concat_1/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=y
&model_1/trans_preprocess/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЖ
 model_1/trans_preprocess/ReshapeReshape*model_1/trans_preprocess/concat_1:output:0/model_1/trans_preprocess/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
 model_1/trans_preprocess/IsNan_2IsNan)model_1/trans_preprocess/Reshape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ|
#model_1/trans_preprocess/LogicalNot
LogicalNot$model_1/trans_preprocess/IsNan_2:y:0*#
_output_shapes
:џџџџџџџџџ
+model_1/trans_preprocess/boolean_mask/ShapeShape)model_1/trans_preprocess/Reshape:output:0*
T0*
_output_shapes
:
9model_1/trans_preprocess/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_1/trans_preprocess/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_1/trans_preprocess/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
3model_1/trans_preprocess/boolean_mask/strided_sliceStridedSlice4model_1/trans_preprocess/boolean_mask/Shape:output:0Bmodel_1/trans_preprocess/boolean_mask/strided_slice/stack:output:0Dmodel_1/trans_preprocess/boolean_mask/strided_slice/stack_1:output:0Dmodel_1/trans_preprocess/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:
<model_1/trans_preprocess/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: и
*model_1/trans_preprocess/boolean_mask/ProdProd<model_1/trans_preprocess/boolean_mask/strided_slice:output:0Emodel_1/trans_preprocess/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 
-model_1/trans_preprocess/boolean_mask/Shape_1Shape)model_1/trans_preprocess/Reshape:output:0*
T0*
_output_shapes
:
;model_1/trans_preprocess/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=model_1/trans_preprocess/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=model_1/trans_preprocess/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5model_1/trans_preprocess/boolean_mask/strided_slice_1StridedSlice6model_1/trans_preprocess/boolean_mask/Shape_1:output:0Dmodel_1/trans_preprocess/boolean_mask/strided_slice_1/stack:output:0Fmodel_1/trans_preprocess/boolean_mask/strided_slice_1/stack_1:output:0Fmodel_1/trans_preprocess/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask
-model_1/trans_preprocess/boolean_mask/Shape_2Shape)model_1/trans_preprocess/Reshape:output:0*
T0*
_output_shapes
:
;model_1/trans_preprocess/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
=model_1/trans_preprocess/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
=model_1/trans_preprocess/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5model_1/trans_preprocess/boolean_mask/strided_slice_2StridedSlice6model_1/trans_preprocess/boolean_mask/Shape_2:output:0Dmodel_1/trans_preprocess/boolean_mask/strided_slice_2/stack:output:0Fmodel_1/trans_preprocess/boolean_mask/strided_slice_2/stack_1:output:0Fmodel_1/trans_preprocess/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask 
5model_1/trans_preprocess/boolean_mask/concat/values_1Pack3model_1/trans_preprocess/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:s
1model_1/trans_preprocess/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : т
,model_1/trans_preprocess/boolean_mask/concatConcatV2>model_1/trans_preprocess/boolean_mask/strided_slice_1:output:0>model_1/trans_preprocess/boolean_mask/concat/values_1:output:0>model_1/trans_preprocess/boolean_mask/strided_slice_2:output:0:model_1/trans_preprocess/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Ш
-model_1/trans_preprocess/boolean_mask/ReshapeReshape)model_1/trans_preprocess/Reshape:output:05model_1/trans_preprocess/boolean_mask/concat:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
5model_1/trans_preprocess/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџб
/model_1/trans_preprocess/boolean_mask/Reshape_1Reshape'model_1/trans_preprocess/LogicalNot:y:0>model_1/trans_preprocess/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:џџџџџџџџџ
+model_1/trans_preprocess/boolean_mask/WhereWhere8model_1/trans_preprocess/boolean_mask/Reshape_1:output:0*'
_output_shapes
:џџџџџџџџџВ
-model_1/trans_preprocess/boolean_mask/SqueezeSqueeze3model_1/trans_preprocess/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
u
3model_1/trans_preprocess/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Й
.model_1/trans_preprocess/boolean_mask/GatherV2GatherV26model_1/trans_preprocess/boolean_mask/Reshape:output:06model_1/trans_preprocess/boolean_mask/Squeeze:output:0<model_1/trans_preprocess/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџk
!model_1/trans_preprocess/Const_19Const*
_output_shapes
:*
dtype0*
valueB: Ћ
model_1/trans_preprocess/MeanMean7model_1/trans_preprocess/boolean_mask/GatherV2:output:0*model_1/trans_preprocess/Const_19:output:0*
T0*
_output_shapes
: k
!model_1/trans_preprocess/Const_20Const*
_output_shapes
:*
dtype0*
valueB: Т
model_1/trans_preprocess/Mean_1Mean7model_1/trans_preprocess/boolean_mask/GatherV2:output:0*model_1/trans_preprocess/Const_20:output:0*
T0*
_output_shapes
:*
	keep_dims(З
model_1/trans_preprocess/sub_23Sub7model_1/trans_preprocess/boolean_mask/GatherV2:output:0(model_1/trans_preprocess/Mean_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ|
model_1/trans_preprocess/SquareSquare#model_1/trans_preprocess/sub_23:z:0*
T0*#
_output_shapes
:џџџџџџџџџk
!model_1/trans_preprocess/Const_21Const*
_output_shapes
:*
dtype0*
valueB: 
model_1/trans_preprocess/Sum_4Sum#model_1/trans_preprocess/Square:y:0*model_1/trans_preprocess/Const_21:output:0*
T0*
_output_shapes
: 
model_1/trans_preprocess/SizeSize7model_1/trans_preprocess/boolean_mask/GatherV2:output:0*
T0*
_output_shapes
: c
!model_1/trans_preprocess/sub_24/yConst*
_output_shapes
: *
dtype0*
value	B :
model_1/trans_preprocess/sub_24Sub&model_1/trans_preprocess/Size:output:0*model_1/trans_preprocess/sub_24/y:output:0*
T0*
_output_shapes
: |
model_1/trans_preprocess/Cast_5Cast#model_1/trans_preprocess/sub_24:z:0*

DstT0*

SrcT0*
_output_shapes
: 
"model_1/trans_preprocess/truediv_5RealDiv'model_1/trans_preprocess/Sum_4:output:0#model_1/trans_preprocess/Cast_5:y:0*
T0*
_output_shapes
: n
model_1/trans_preprocess/SqrtSqrt&model_1/trans_preprocess/truediv_5:z:0*
T0*
_output_shapes
: А
model_1/trans_preprocess/sub_25Sub*model_1/trans_preprocess/concat_1:output:0&model_1/trans_preprocess/Mean:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=Ћ
"model_1/trans_preprocess/truediv_6RealDiv#model_1/trans_preprocess/sub_25:z:0!model_1/trans_preprocess/Sqrt:y:0*
T0*+
_output_shapes
:џџџџџџџџџ=
/model_1/trans_preprocess/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1model_1/trans_preprocess/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1model_1/trans_preprocess/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
)model_1/trans_preprocess/strided_slice_25StridedSlice&model_1/trans_preprocess/truediv_6:z:08model_1/trans_preprocess/strided_slice_25/stack:output:0:model_1/trans_preprocess/strided_slice_25/stack_1:output:0:model_1/trans_preprocess/strided_slice_25/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*
end_mask
#model_1/trans_preprocess/zeros_like	ZerosLike2model_1/trans_preprocess/strided_slice_25:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=y
/model_1/trans_preprocess/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB: 
1model_1/trans_preprocess/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1model_1/trans_preprocess/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
)model_1/trans_preprocess/strided_slice_26StridedSlice&model_1/trans_preprocess/truediv_6:z:08model_1/trans_preprocess/strided_slice_26/stack:output:0:model_1/trans_preprocess/strided_slice_26/stack_1:output:0:model_1/trans_preprocess/strided_slice_26/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*

begin_masky
/model_1/trans_preprocess/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1model_1/trans_preprocess/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1model_1/trans_preprocess/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
)model_1/trans_preprocess/strided_slice_27StridedSlice&model_1/trans_preprocess/truediv_6:z:08model_1/trans_preprocess/strided_slice_27/stack:output:0:model_1/trans_preprocess/strided_slice_27/stack_1:output:0:model_1/trans_preprocess/strided_slice_27/stack_2:output:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ=*
end_maskФ
model_1/trans_preprocess/sub_26Sub2model_1/trans_preprocess/strided_slice_26:output:02model_1/trans_preprocess/strided_slice_27:output:0*
T0*+
_output_shapes
:џџџџџџџџџ=h
&model_1/trans_preprocess/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
!model_1/trans_preprocess/concat_2ConcatV2#model_1/trans_preprocess/sub_26:z:0'model_1/trans_preprocess/zeros_like:y:0/model_1/trans_preprocess/concat_2/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=~
model_1/trans_preprocess/NegNeg#model_1/trans_preprocess/sub_26:z:0*
T0*+
_output_shapes
:џџџџџџџџџ=h
&model_1/trans_preprocess/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
!model_1/trans_preprocess/concat_3ConcatV2'model_1/trans_preprocess/zeros_like:y:0 model_1/trans_preprocess/Neg:y:0/model_1/trans_preprocess/concat_3/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ=y
(model_1/trans_preprocess/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   У
*model_1/trans_preprocess/flatten_1/ReshapeReshape&model_1/trans_preprocess/truediv_6:z:01model_1/trans_preprocess/flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗ{
*model_1/trans_preprocess/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   Ы
,model_1/trans_preprocess/flatten_1/Reshape_1Reshape*model_1/trans_preprocess/concat_2:output:03model_1/trans_preprocess/flatten_1/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗ{
*model_1/trans_preprocess/flatten_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"џџџџЗ   Ы
,model_1/trans_preprocess/flatten_1/Reshape_2Reshape*model_1/trans_preprocess/concat_3:output:03model_1/trans_preprocess/flatten_1/Const_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџЗ{
*model_1/trans_preprocess/flatten_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"џџџџО   б
,model_1/trans_preprocess/flatten_1/Reshape_3Reshape0model_1/trans_preprocess/norm_1/Squeeze:output:03model_1/trans_preprocess/flatten_1/Const_3:output:0*
T0*(
_output_shapes
:џџџџџџџџџО{
*model_1/trans_preprocess/flatten_1/Const_4Const*
_output_shapes
:*
dtype0*
valueB"џџџџв   Я
,model_1/trans_preprocess/flatten_1/Reshape_4Reshape.model_1/trans_preprocess/norm/Squeeze:output:03model_1/trans_preprocess/flatten_1/Const_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџв{
*model_1/trans_preprocess/flatten_1/Const_5Const*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ц
,model_1/trans_preprocess/flatten_1/Reshape_5Reshape&model_1/trans_preprocess/truediv_4:z:03model_1/trans_preprocess/flatten_1/Const_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ{
*model_1/trans_preprocess/flatten_1/Const_6Const*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ц
,model_1/trans_preprocess/flatten_1/Reshape_6Reshape&model_1/trans_preprocess/truediv_2:z:03model_1/trans_preprocess/flatten_1/Const_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџq
&model_1/trans_preprocess/concat_4/axisConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
!model_1/trans_preprocess/concat_4ConcatV23model_1/trans_preprocess/flatten_1/Reshape:output:05model_1/trans_preprocess/flatten_1/Reshape_1:output:05model_1/trans_preprocess/flatten_1/Reshape_2:output:05model_1/trans_preprocess/flatten_1/Reshape_3:output:05model_1/trans_preprocess/flatten_1/Reshape_4:output:05model_1/trans_preprocess/flatten_1/Reshape_5:output:05model_1/trans_preprocess/flatten_1/Reshape_6:output:0/model_1/trans_preprocess/concat_4/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџЬ
 model_1/trans_preprocess/IsNan_3IsNan*model_1/trans_preprocess/concat_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬj
%model_1/trans_preprocess/SelectV2_1/tConst*
_output_shapes
: *
dtype0*
valueB
 *    ф
#model_1/trans_preprocess/SelectV2_1SelectV2$model_1/trans_preprocess/IsNan_3:y:0.model_1/trans_preprocess/SelectV2_1/t:output:0*model_1/trans_preprocess/concat_4:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬy
/model_1/trans_preprocess/strided_slice_28/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_1/trans_preprocess/strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1model_1/trans_preprocess/strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ђ
)model_1/trans_preprocess/strided_slice_28StridedSlice,model_1/trans_preprocess/SelectV2_1:output:08model_1/trans_preprocess/strided_slice_28/stack:output:0:model_1/trans_preprocess/strided_slice_28/stack_1:output:0:model_1/trans_preprocess/strided_slice_28/stack_2:output:0*
Index0*
T0*,
_output_shapes
:џџџџџџџџџЬ*
new_axis_mask
IdentityIdentity2model_1/trans_preprocess/strided_slice_28:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
№
G
!__inference__traced_restore_98086
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ѓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ь
P
'__inference_model_1_layer_call_fn_97577

inputs
unknown
identityМ
PartitionedCallPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_97572e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a
ь
P
'__inference_model_1_layer_call_fn_97613

inputs
unknown
identityМ
PartitionedCallPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџЬ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_97601e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:џџџџџџџџџЬ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџ:a:T P
,
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs: 

_output_shapes
:a"
J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_default
>
inputs4
serving_default_inputs:0џџџџџџџџџA
trans_preprocess-
PartitionedCall:0џџџџџџџџџЬtensorflow/serving/predict:О;
ю
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures"
_tf_keras_network
"
_tf_keras_input_layer
В
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
flatten"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ы
trace_0
trace_12
'__inference_model_1_layer_call_fn_97577
'__inference_model_1_layer_call_fn_97613П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12Ъ
B__inference_model_1_layer_call_and_return_conditional_losses_97620
B__inference_model_1_layer_call_and_return_conditional_losses_97627П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ш
	capture_0BЧ
 __inference__wrapped_model_97151inputs"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0
,
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
"trace_02д
0__inference_trans_preprocess_layer_call_fn_97643
В
FullArgSpec
args
jself
jpos
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z"trace_0

#trace_02я
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_98052
В
FullArgSpec
args
jself
jpos
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z#trace_0
Ѕ
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper

	capture_0Bѕ
'__inference_model_1_layer_call_fn_97577inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0

	capture_0Bѕ
'__inference_model_1_layer_call_fn_97613inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0
Б
	capture_0B
B__inference_model_1_layer_call_and_return_conditional_losses_97620inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0
Б
	capture_0B
B__inference_model_1_layer_call_and_return_conditional_losses_97627inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0
J
Constjtf.TrackableConstant
ч
	capture_0BЦ
#__inference_signature_wrapper_97636inputs"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ќ
	capture_0Bл
0__inference_trans_preprocess_layer_call_fn_97643pos"
В
FullArgSpec
args
jself
jpos
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0

	capture_0Bі
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_98052pos"
В
FullArgSpec
args
jself
jpos
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z	capture_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapperЈ
 __inference__wrapped_model_971514Ђ1
*Ђ'
%"
inputsџџџџџџџџџ
Њ "HЊE
C
trans_preprocess/,
trans_preprocessџџџџџџџџџЬК
B__inference_model_1_layer_call_and_return_conditional_losses_97620t<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ
p 

 
Њ "1Ђ.
'$
tensor_0џџџџџџџџџЬ
 К
B__inference_model_1_layer_call_and_return_conditional_losses_97627t<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ
p

 
Њ "1Ђ.
'$
tensor_0џџџџџџџџџЬ
 
'__inference_model_1_layer_call_fn_97577i<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ
p 

 
Њ "&#
unknownџџџџџџџџџЬ
'__inference_model_1_layer_call_fn_97613i<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ
p

 
Њ "&#
unknownџџџџџџџџџЬЕ
#__inference_signature_wrapper_97636>Ђ;
Ђ 
4Њ1
/
inputs%"
inputsџџџџџџџџџ"HЊE
C
trans_preprocess/,
trans_preprocessџџџџџџџџџЬИ
K__inference_trans_preprocess_layer_call_and_return_conditional_losses_98052i1Ђ.
'Ђ$
"
posџџџџџџџџџ
Њ "1Ђ.
'$
tensor_0џџџџџџџџџЬ
 
0__inference_trans_preprocess_layer_call_fn_97643^1Ђ.
'Ђ$
"
posџџџџџџџџџ
Њ "&#
unknownџџџџџџџџџЬ