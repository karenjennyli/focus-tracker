import {
    Box,
    Container,
    Flex,
    Heading,
    Image,
    Text,
  } from '@chakra-ui/react';
import emotiveHeadset from './emotiveHeadset.png'
  
function EmotiveHeadsetComponent() {
    return (
        <Container maxW="container.xl" py={20}>
            <Flex direction={{ base: "column", lg: "row" }} align="center" justify="space-between" gap={20}>
                
                {/* Left side - Text content */}
                <Box flex="1" color="white">
                    <Heading as="h1" size="2xl" fontWeight="bold" mb={4} sx={{textShadow: `-1px -1px 0 green, 1px -1px 0 green, -1px 1px 0 green, 1px 1px 0 green`}}>
                        Emotiv Insight EEG Headset Technology
                    </Heading>
                    <Text fontSize="lg" mb={6}>
                        The Insight headset has 5 sensors which places it right in the middle of other EEG headset options which have anywhere from 1-32 sensors to measure EEG signals from different regions of the brain.
                    </Text>
                </Box>

                <Box position="relative" width={{ base: "100%", lg: "30%" }} mr={{ lg: "20%", xl: "20%" }}>
                    <Image
                        src={emotiveHeadset}
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

export default EmotiveHeadsetComponent;
