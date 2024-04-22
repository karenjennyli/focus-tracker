import {
    Box,
    Button,
    Container,
    Flex,
    Heading,
    Image,
    Text,
  } from '@chakra-ui/react';
import {useNavigate} from 'react-router-dom';
import currentSession from './current_session_image.png'
import sessionHistory from './session_history_image.png'
  
function FeatureComponent() {
    const navigate = useNavigate();

    return (
        <Container maxW="container.xl" py={14}>
        <Flex direction={{ base: "column", lg: "row" }} align="center" justify="space-between" gap={20}>
            
            {/* Left side - Text content */}
            <Box flex="1" color="white">
            <Heading as="h1" size="3xl" fontWeight="bold" mb={4} sx={{textShadow: `-1px -1px 0 red, 1px -1px 0 red, -1px 1px 0 red, 1px 1px 0 red`}}>
                Real-Time data visualization and analytics
            </Heading>
            <Text fontSize="lg" mb={6}>
                Immediate distraction feedback and interactive graphs provided
            </Text>
            <Button colorScheme="blue" size="lg" px={10} py={6} onClick={() => navigate('/calibration-page')}>
                Get started
            </Button>
            </Box>

            {/* Right side - Overlapping images */}
            <Box position="relative" width={{ base: "100%", lg: "30%" }} mr={{ lg: "20%", xl: "20%" }}>
            <Image
                src={currentSession}
                alt="Illustration of connection"
                position="relative"
                zIndex="1"
                border='2px solid white'
            />
            <Image
                src={sessionHistory}
                alt="Illustration of transaction"
                position="absolute"
                right="-80%"
                top="50%"
                transform="translateY(-10%)"
                zIndex="0"
                width="100%" // Used to adjust the width
                border='2px solid white'
            />
            </Box>

        </Flex>
        </Container>
    );
}

export default FeatureComponent;
