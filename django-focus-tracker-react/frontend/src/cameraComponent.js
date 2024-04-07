import {
    Box,
    Container,
    Flex,
    Heading,
    Image,
    Text,
  } from '@chakra-ui/react';
  import camera from './camera2.png'
  
function CameraCompontent() {
    return (
        <Container maxW="container.xl">
            <Flex direction={{ base: "column", lg: "row" }} align="center" justify="space-between" gap={20}>
                
                {/* Left side - Text content */}
                <Box flex="1" color="white">
                    <Heading as="h1" size="3xl" fontWeight="bold" mb={4}>
                        Camera-Based Detection Technology
                    </Heading>
                    <Text fontSize="lg" mb={6}>
                        The TedGem 1080p camera has a processing rate 10 fps and ensures high-quality real-time monitoring of physical indicators of loss of focus, such as yawning, microsleeps, off-screen gazing, interruptions, and phone pick-ups. 
                    </Text>
                </Box>

                <Box position="relative" width={{ base: "100%", lg: "30%" }} mr={{ lg: "20%", xl: "20%" }}>
                    <Image
                        src={camera}
                        alt="Illustration of connection"
                        position="relative"
                        zIndex="1"
                        right="-40%"
                    />
                </Box>

            </Flex>
        </Container>
    );
}

export default CameraCompontent;
