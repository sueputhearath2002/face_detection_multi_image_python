# import 'dart:io';
# import 'dart:ui' as ui;

# import 'package:flutter/material.dart';
# import 'package:flutter/services.dart';
# import 'package:google_ml_kit/google_ml_kit.dart';
# import 'package:image/image.dart' as img;
# import 'package:image_picker/image_picker.dart';
# import 'package:path_provider/path_provider.dart';
# import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
# import 'package:tomnenh/helper/helper.dart';
# import 'package:tomnenh/screen/uploads/upload_cubit.dart';
# import 'package:tomnenh/style/assets.dart';
# import 'package:tomnenh/style/colors.dart';
# import 'package:tomnenh/widget/app_bar_custom_simple.dart'
#     show AppbarCustomSimple;
# import 'package:tomnenh/widget/elevated_btn_cus.dart';
# import 'package:tomnenh/widget/text_widget.dart';
# import 'package:tomnenh/widget/upload_image.dart';

# class UploadFaceDetectionScreen extends StatefulWidget {
#   const UploadFaceDetectionScreen({super.key});
#   static const String routeName = "face_detection";

#   @override
#   State<UploadFaceDetectionScreen> createState() =>
#       _UploadFaceDetectionScreenState();
# }

# class _UploadFaceDetectionScreenState extends State<UploadFaceDetectionScreen> {
#   File? _image;
#   ui.Image? _displayImage;
#   List<Face> _detectedFaces = [];
#   String? _predictedLabel;
#   tfl.Interpreter? _interpreter;
#   List<String> _labels = [];
#   final ImagePicker _picker = ImagePicker();
#   final faceDetector = GoogleMlKit.vision.faceDetector();
#   final screenCubit = UploadCubit();
#   List<String> listLabels = [];

#   @override
#   void initState() {
#     super.initState();
#     _loadModel();
#     _loadLabels();
#   }

#   Future<void> checkAttendance() async {
#     final names = listLabels.map((e) => e.split(' (').first).toList();
#     final result = await screenCubit.checkAttendance(names);
#     if (result) {
#       Helper.showMessage(msg: "Attendance submitted successfully!");
#       if (!mounted) return;
#       Navigator.pop(context);
#     }
#   }

#   Future<void> _loadModel() async {
#     try {
#       File? tLifeFile = await Helper.getLocalModelFile('face_detection.tflite');
#       print("===========file: ${tLifeFile}");
#       if (tLifeFile != null) {
#         final exists = await tLifeFile.exists();
#         print("File exists: $exists");
#         _interpreter = await tfl.Interpreter.fromFile(tLifeFile);
#         print("Model loaded!");
#         print(
#             "Interpreter input tensor count: ${_interpreter!.allocateTensors()}");
#         print(
#             "Interpreter output tensor count: ${_interpreter!.getOutputTensorCount()}");
#       } else {
#         print("Model not found. Please download it first.");
#       }
#     } catch (e) {
#       debugPrint("Error loading model: $e");
#     }
#   }

#   Future<void> _loadLabels() async {
#     try {
#       File? labelFile = await Helper.getLocalModelFile('labels.txt');
#       if (labelFile != null && await labelFile.exists()) {
#         String labelData = await labelFile.readAsString();
#         setState(() {
#           _labels =
#               labelData.split("\n").where((label) => label.isNotEmpty).toList();
#         });
#       } else {
#         debugPrint("Label file not found.");
#       }
#     } catch (e) {
#       debugPrint("Error loading labels: $e");
#     }
#   }

#   Future<void> _pickImage(ImageSource source) async {
#     Navigator.pop(context);
#     final pickedFile = await _picker.pickImage(source: source);
#     if (pickedFile != null) {
#       File imageFile = File(pickedFile.path);
#       setState(() => _image = imageFile);
#       listLabels.clear();
#       _predictImage(imageFile);
#     }
#   }

#   Future<void> _predictImage(File image) async {
#     if (_interpreter == null || _labels.isEmpty) {
#       setState(() {
#         _predictedLabel = "Model or labels not loaded.";
#         _displayImage = null;
#       });
#       return;
#     }

#     _detectedFaces =
#         await faceDetector.processImage(InputImage.fromFile(image));

#     if (_detectedFaces.isEmpty) {
#       setState(() {
#         _predictedLabel = "No faces detected";
#         _displayImage = null;
#       });
#       return;
#     }

#     listLabels.clear(); // Clear previous predictions
#     Set<String> uniqueNames = {}; // For checking duplicates
#     List<String> detectedLabels = [];

#     for (final face in _detectedFaces) {
#       final croppedFace = await _cropFace(image, face);
#       final input = await _processImage(croppedFace);
#       final output = List.generate(1, (_) => List.filled(_labels.length, 0.0));

#       _interpreter!.run(input, output);
#       final scores = output[0];

#       for (int i = 0; i < scores.length; i++) {
#         final confidence = scores[i];
#         if (confidence > 0.5) {
#           // Adjusted confidence threshold
#           final name = _labels[i];
#           if (!uniqueNames.contains(name)) {
#             final label = "$name (${(confidence * 100).toStringAsFixed(1)}%)";
#             uniqueNames.add(name);
#             detectedLabels.add(label);
#           }
#         }
#       }
#     }

#     final imageBytes = await image.readAsBytes();
#     final codec = await ui.instantiateImageCodec(imageBytes);
#     final frame = await codec.getNextFrame();

#     setState(() {
#       _displayImage = frame.image;
#       _predictedLabel = detectedLabels.isEmpty
#           ? "No faces confidently detected"
#           : detectedLabels.join("\n");
#       listLabels.addAll(detectedLabels);
#     });

#     print("Detected Labels: $listLabels");
#   }

#   // Future<void> _predictImage(File image) async {
#   //   if (_interpreter == null || _labels.isEmpty) return;
#   //   _detectedFaces =
#   //       await faceDetector.processImage(InputImage.fromFile(image));
#   //   if (_detectedFaces.isEmpty) {
#   //     setState(() => _predictedLabel = "No faces detected");
#   //     return;
#   //   }
#   //
#   //   List<String> detectedFaces = [];
#   //   for (var face in _detectedFaces) {
#   //     var croppedFace = await _cropFace(image, face);
#   //     var input = await _processImage(croppedFace);
#   //     var output = List.generate(1, (_) => List.filled(_labels.length, 0.0));
#   //     _interpreter?.run(input, output);
#   //     List<double> outputList = List<double>.from(output[0]);
#   //     for (int i = 0; i < outputList.length; i++) {
#   //       if (outputList[i] > 0.1) {
#   //         detectedFaces.add(
#   //             "${_labels[i]} (${(outputList[i] * 100).toStringAsFixed(1)}%)");
#   //       }
#   //     }
#   //   }
#   //
#   //   final imgData = await image.readAsBytes();
#   //   final codec = await ui.instantiateImageCodec(imgData);
#   //   final frame = await codec.getNextFrame();
#   //   setState(() {
#   //     _displayImage = frame.image;
#   //     _predictedLabel = detectedFaces.isEmpty
#   //         ? "No faces confidently detected"
#   //         : detectedFaces.join("\n");
#   //     listLabels.addAll(detectedFaces);
#   //   });
#   //   print("===============asda=================$listLabels");
#   // }

#   Future<File> _cropFace(File image, Face face) async {
#     final Uint8List imageData = await image.readAsBytes();
#     img.Image? rawImage = img.decodeImage(imageData);
#     if (rawImage == null) throw Exception("Error decoding image");

#     final rect = face.boundingBox;
#     int x = (rect.left - 20).toInt().clamp(0, rawImage.width);
#     int y = (rect.top - 20).toInt().clamp(0, rawImage.height);
#     int width = (rect.width + 40).toInt().clamp(0, rawImage.width - x);
#     int height = (rect.height + 40).toInt().clamp(0, rawImage.height - y);
#     img.Image cropped =
#         img.copyCrop(rawImage, x: x, y: y, width: width, height: height);

#     final tempPath = '${(await getTemporaryDirectory()).path}/cropped_face.png';
#     return File(tempPath)..writeAsBytesSync(img.encodePng(cropped));
#   }

#   Future<List<List<List<List<double>>>>> _processImage(File image) async {
#     final Uint8List imageData = await image.readAsBytes();
#     img.Image? rawImage = img.decodeImage(imageData);
#     if (rawImage == null) throw Exception("Error decoding image");

#     img.Image resizedImage = img.copyResize(rawImage, width: 224, height: 224);
#     List<List<List<double>>> inputImage = List.generate(
#       224,
#       (y) => List.generate(
#         224,
#         (x) {
#           var pixel = resizedImage.getPixel(x, y);
#           return [
#             (pixel.r - 128) / 128,
#             (pixel.g - 128) / 128,
#             (pixel.b - 128) / 128,
#           ];
#         },
#       ),
#     );
#     return [inputImage];
#   }

#   @override
#   void dispose() {
#     _interpreter?.close();
#     faceDetector.close();
#     super.dispose();
#   }

#   @override
#   Widget build(BuildContext context) {
#     return Scaffold(
#       appBar: const AppbarCustomSimple(title: "Upload Face Detection"),
#       persistentFooterButtons: [
#         ElevatedBtnCus(
#           onTap: () => checkAttendance(),
#           isFullWidth: true,
#           btnName: "Submit Attendance",
#         ),
#       ],
#       body: SingleChildScrollView(
#         child: Padding(
#           padding: const EdgeInsets.symmetric(horizontal: 16),
#           child: Column(
#             spacing: 8,
#             mainAxisAlignment: MainAxisAlignment.start,
#             children: [
#               const Align(
#                 alignment: Alignment.topLeft,
#                 child: TextWidget(
#                   text: "Note: Upload image by Tap below.",
#                   color: mainColor,
#                 ),
#               ),
#               Stack(
#                 children: [
#                   UploadImage(imgFile: _image),
#                   if (_displayImage != null)
#                     Positioned.fill(
#                       child: CustomPaint(
#                         painter: FacePainter(
#                           faces: _detectedFaces,
#                           image: _displayImage!,
#                         ),
#                       ),
#                     ),
#                 ],
#               ),
#               ElevatedBtnCus(
#                 onTap: () {
#                   showModalBottomSheet<void>(
#                     context: context,
#                     builder: (BuildContext context) {
#                       return buildOptionSelectPicture();
#                     },
#                   );
#                 },
#                 icon: Icons.image,
#                 btnName: "Upload Image",
#               ),
#               const Align(
#                 alignment: Alignment.topLeft,
#                 child: TextWidget(
#                   text: "List of Student",
#                   fontSize: 16,
#                   fontWeight: FontWeight.bold,
#                 ),
#               ),
#               listLabels.isEmpty
#                   ? Padding(
#                       padding: const EdgeInsets.only(top: 16),
#                       child: Column(
#                         crossAxisAlignment: CrossAxisAlignment.center,
#                         children: [
#                           Image.asset(noDataPng, height: 300),
#                           const Text("Data not found yet!",
#                               style: TextStyle(fontSize: 16)),
#                         ],
#                       ),
#                     )
#                   : Column(
#                       crossAxisAlignment: CrossAxisAlignment.start,
#                       children: List.generate(
#                         listLabels.length,
#                         (int index) {
#                           return Row(
#                             children: [
#                               const Icon(Icons.check_circle,
#                                   color: Colors.green, size: 18),
#                               const SizedBox(width: 8),
#                               Text(
#                                 RegExp(r'^(.*?)\s*\(')
#                                         .firstMatch(listLabels[index])
#                                         ?.group(1) ??
#                                     '',
#                                 style: const TextStyle(fontSize: 16),
#                               ),
#                             ],
#                           );
#                         },
#                       ),
#                     ),
#             ],
#           ),
#         ),
#       ),
#     );
#   }

#   Widget buildOptionSelectPicture() {
#     return Container(
#       padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
#       width: double.infinity,
#       child: SafeArea(
#         child: Column(
#           mainAxisSize: MainAxisSize.min,
#           children: [
#             const ListTile(
#               title: TextWidget(
#                 text: "Let's pick image from option below.",
#                 fontSize: 14,
#                 fontWeight: FontWeight.bold,
#               ),
#               leading: Icon(Icons.info, color: mainColor),
#             ),
#             ListTile(
#               leading: const Icon(Icons.photo_library),
#               title: const Text("Gallery"),
#               onTap: () => _pickImage(ImageSource.gallery),
#             ),
#             ListTile(
#               leading: const Icon(Icons.camera_alt),
#               title: const Text("Camera"),
#               onTap: () => _pickImage(ImageSource.camera),
#             ),
#           ],
#         ),
#       ),
#     );
#   }
# }

